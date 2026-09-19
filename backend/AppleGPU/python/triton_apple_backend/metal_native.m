// metal_native - the runtime for installs without torch. Owns its device and
// command queue; pointer arguments are MetalBuffer objects, from `alloc` or
// `wrap` over caller memory.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <deque>
#include <string.h>
#include <vector>

// Undeclared in the SDK headers, so the call site would otherwise infer `id`.
@protocol MTLLibraryDataContents
- (dispatch_data_t)libraryDataContents;
@end

// Read and written under the GIL. `sync_gpu` is the one place that drops it,
// and it takes `g_last` before releasing and reaps after reacquiring.
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLCommandBuffer> g_last = nil;

// A committed command buffer keeps Python references to its MetalBuffer
// arguments until it is seen complete, so wrapped caller memory cannot be
// released under the GPU. Entries complete in commit order.
struct Inflight {
  id<MTLCommandBuffer> cb;
  std::vector<PyObject *> held;
};
static std::deque<Inflight> g_inflight;

static id<MTLDevice> get_device(void) {
  if (!g_device) {
    g_device = MTLCreateSystemDefaultDevice();
    g_queue = [g_device newCommandQueue];
  }
  return g_device;
}

// The watchdog aborts asynchronously, so a reaped failure is held for the
// next synchronize to raise.
static NSError *g_reapError = nil;

// GIL held.
static void reapInflight(void) {
  while (!g_inflight.empty()) {
    const MTLCommandBufferStatus st = g_inflight.front().cb.status;
    if (st != MTLCommandBufferStatusCompleted &&
        st != MTLCommandBufferStatusError)
      break;
    if (st == MTLCommandBufferStatusError && !g_reapError)
      g_reapError = g_inflight.front().cb.error;
    for (PyObject *o : g_inflight.front().held)
      Py_DECREF(o);
    g_inflight.pop_front();
  }
}

// Command buffers on one queue complete in commit order, so waiting on the
// last one drains the queue. False with the Python error set.
static bool sync_gpu(void) {
  id<MTLCommandBuffer> cb = g_last;
  g_last = nil;
  if (!cb)
    return true;
  Py_BEGIN_ALLOW_THREADS;
  [cb waitUntilCompleted];
  Py_END_ALLOW_THREADS;
  reapInflight();
  NSError *failed = cb.error ? cb.error : g_reapError;
  if (failed) {
    PyErr_Format(PyExc_RuntimeError, "Metal command buffer failed: %s",
                 failed.localizedDescription.UTF8String);
    g_reapError = nil;
    return false;
  }
  return true;
}

static void appendErrorDetail(NSMutableString *full, NSError *error) {
  [full appendFormat:@"%@", [error localizedDescription]];
  NSDictionary *info = [error userInfo];
  if (info && [info count])
    [full appendFormat:@" | userInfo=%@", info];
  NSError *under = [[error userInfo] objectForKey:NSUnderlyingErrorKey];
  if (under)
    [full appendFormat:@" | underlying=%@ (%@)", [under localizedDescription],
                       [under userInfo]];
}

// ── MetalBuffer ──────────────────────────────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLBuffer> buf;
  Py_buffer view;
  bool hasView;
  PyObject *dtype;
  Py_ssize_t nbytes;
} MetalBufferObject;

static void MetalBuffer_dealloc(MetalBufferObject *self) {
  self->buf = nil;
  if (self->hasView)
    PyBuffer_Release(&self->view);
  Py_CLEAR(self->dtype);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MetalBuffer_data_ptr(MetalBufferObject *self,
                                      PyObject *Py_UNUSED(args)) {
  return PyLong_FromVoidPtr([self->buf contents]);
}

// The address a shader dereferences: `contents` is the CPU mapping.
static PyObject *MetalBuffer_gpu_address(MetalBufferObject *self,
                                         PyObject *Py_UNUSED(args)) {
  // A no-copy view over host memory reads as zeros through a raw address,
  // whether or not it is also bound.
  if (self->hasView) {
    PyErr_SetString(PyExc_RuntimeError,
                    "a wrapped buffer has no address a kernel can read "
                    "through; allocate with metal_native.alloc instead");
    return NULL;
  }
  return PyLong_FromUnsignedLongLong([self->buf gpuAddress]);
}

static PyObject *MetalBuffer_get_dtype(MetalBufferObject *self, void *closure) {
  PyObject *d = self->dtype ? self->dtype : Py_None;
  Py_INCREF(d);
  return d;
}

static PyObject *MetalBuffer_get_nbytes(MetalBufferObject *self,
                                        void *closure) {
  return PyLong_FromSsize_t(self->nbytes);
}

static int MetalBuffer_getbuffer(MetalBufferObject *self, Py_buffer *view,
                                 int flags) {
  if (!sync_gpu()) {
    view->obj = NULL;
    return -1;
  }
  return PyBuffer_FillInfo(view, (PyObject *)self, [self->buf contents],
                           self -> nbytes, 0, flags);
}

static PyBufferProcs MetalBuffer_as_buffer = {
    (getbufferproc)MetalBuffer_getbuffer,
    NULL,
};

static PyMethodDef MetalBuffer_methods[] = {
    {"data_ptr", (PyCFunction)MetalBuffer_data_ptr, METH_NOARGS, NULL},
    {"gpu_address", (PyCFunction)MetalBuffer_gpu_address, METH_NOARGS, NULL},
    {NULL}};

static PyGetSetDef MetalBuffer_getset[] = {
    {"dtype", (getter)MetalBuffer_get_dtype, NULL, NULL, NULL},
    {"nbytes", (getter)MetalBuffer_get_nbytes, NULL, NULL, NULL},
    {NULL}};

static PyTypeObject MetalBufferType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "metal_native.MetalBuffer",
    .tp_basicsize = sizeof(MetalBufferObject),
    .tp_dealloc = (destructor)MetalBuffer_dealloc,
    .tp_as_buffer = &MetalBuffer_as_buffer,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = MetalBuffer_methods,
    .tp_getset = MetalBuffer_getset,
};

// tp_alloc zero-fills, which ARC needs before the first assignment to an
// object slot (it releases the previous value).
static MetalBufferObject *newMetalBuffer(id<MTLBuffer> buf, Py_ssize_t nbytes,
                                         PyObject *dtype) {
  MetalBufferObject *obj =
      (MetalBufferObject *)MetalBufferType.tp_alloc(&MetalBufferType, 0);
  if (!obj)
    return NULL;
  obj->buf = buf;
  obj->nbytes = nbytes;
  obj->dtype = (dtype && dtype != Py_None) ? Py_NewRef(dtype) : NULL;
  return obj;
}

static PyObject *py_alloc(PyObject *self, PyObject *args) {
  Py_ssize_t nbytes;
  PyObject *dtype = Py_None;
  if (!PyArg_ParseTuple(args, "n|O", &nbytes, &dtype))
    return NULL;
  if (nbytes < 0) {
    PyErr_SetString(PyExc_ValueError, "nbytes must be non-negative");
    return NULL;
  }
  @autoreleasepool {
    // Metal refuses a zero-length buffer.
    NSUInteger len = nbytes > 0 ? (NSUInteger)nbytes : 16;
    id<MTLBuffer> buf =
        [get_device() newBufferWithLength:len
                                  options:MTLResourceStorageModeShared];
    if (!buf) {
      PyErr_Format(PyExc_MemoryError, "Metal could not allocate %zd bytes",
                   nbytes);
      return NULL;
    }
    memset([buf contents], 0, len);
    return (PyObject *)newMetalBuffer(buf, nbytes, dtype);
  }
}

static PyObject *py_wrap(PyObject *self, PyObject *args) {
  PyObject *obj;
  PyObject *dtype = Py_None;
  if (!PyArg_ParseTuple(args, "O|O", &obj, &dtype))
    return NULL;

  MetalBufferObject *out = newMetalBuffer(nil, 0, NULL);
  if (!out)
    return NULL;
  if (PyObject_GetBuffer(obj, &out->view, PyBUF_SIMPLE) < 0) {
    Py_DECREF(out);
    return NULL;
  }
  out->hasView = true;
  out->nbytes = out->view.len;
  if (out->view.len == 0) {
    Py_DECREF(out);
    PyErr_SetString(PyExc_ValueError, "cannot wrap an empty buffer");
    return NULL;
  }
  @autoreleasepool {
    out->buf =
        [get_device() newBufferWithBytesNoCopy:out->view.buf
                                        length:(NSUInteger)out->view.len
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
  }
  if (!out->buf) {
    Py_DECREF(out);
    PyErr_Format(PyExc_ValueError,
                 "Metal declined a zero-copy view of this memory (%p, %zd "
                 "bytes); the documented requirement is page alignment. Copy "
                 "into alloc() instead.",
                 out->view.buf, out->view.len);
    return NULL;
  }
  if (dtype != Py_None) {
    out->dtype = Py_NewRef(dtype);
  } else if (PyObject_HasAttrString(obj, "dtype")) {
    out->dtype = PyObject_GetAttrString(obj, "dtype");
    if (!out->dtype) {
      Py_DECREF(out);
      return NULL;
    }
  }
  return (PyObject *)out;
}

// ── MetalKernel ──────────────────────────────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLComputePipelineState> pso;
  NSUInteger maxThreads;
  PyObject *name;
} MetalKernelObject;

static void MetalKernel_dealloc(MetalKernelObject *self) {
  self->pso = nil;
  Py_CLEAR(self->name);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

struct ArgInfo {
  enum Kind { BUFFER, INT, FLOAT, BYTES } kind;
  id<MTLBuffer> buf;
  PyObject *obj;
  int64_t intVal;
  float floatVal;
  const void *bytesPtr;
  Py_ssize_t bytesLen;
};

struct LaunchGeometry {
  long tx, ty, tz;
  long gx, gy, gz;
  long tgmem;
};

static bool readLaunchGeometry(PyObject *kwargs, LaunchGeometry *out) {
  PyObject *threads_obj = NULL, *group_obj = NULL, *tgmem_obj = NULL;
  if (kwargs) {
    threads_obj = PyDict_GetItemString(kwargs, "threads");
    group_obj = PyDict_GetItemString(kwargs, "group_size");
    tgmem_obj = PyDict_GetItemString(kwargs, "threadgroup_mem");
  }
  if (!threads_obj || !group_obj) {
    PyErr_SetString(PyExc_ValueError, "threads and group_size required");
    return false;
  }
  out->tgmem = tgmem_obj ? PyLong_AsLong(tgmem_obj) : 0;
  out->tx = PyLong_AsLong(PyList_GetItem(threads_obj, 0));
  out->ty = PyLong_AsLong(PyList_GetItem(threads_obj, 1));
  out->tz = PyLong_AsLong(PyList_GetItem(threads_obj, 2));
  out->gx = PyLong_AsLong(PyList_GetItem(group_obj, 0));
  out->gy = PyLong_AsLong(PyList_GetItem(group_obj, 1));
  out->gz = PyLong_AsLong(PyList_GetItem(group_obj, 2));
  return !PyErr_Occurred();
}

static bool packArguments(PyObject *args, std::vector<ArgInfo> *out) {
  const Py_ssize_t nargs = PyTuple_Size(args);
  out->resize(nargs);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject *arg = PyTuple_GetItem(args, i);
    ArgInfo &info = (*out)[i];
    if (PyObject_TypeCheck(arg, &MetalBufferType)) {
      info.kind = ArgInfo::BUFFER;
      info.buf = ((MetalBufferObject *)arg)->buf;
      info.obj = arg;
    } else if (PyBytes_Check(arg)) {
      info.kind = ArgInfo::BYTES;
      info.bytesPtr = PyBytes_AS_STRING(arg);
      info.bytesLen = PyBytes_GET_SIZE(arg);
    } else if (PyLong_Check(arg)) {
      info.kind = ArgInfo::INT;
      info.intVal = PyLong_AsLongLong(arg);
    } else if (PyFloat_Check(arg)) {
      info.kind = ArgInfo::FLOAT;
      info.floatVal = (float)PyFloat_AsDouble(arg);
    } else {
      PyErr_Format(PyExc_TypeError,
                   "Arg %zd: expected MetalBuffer, int, float, or bytes (got "
                   "%s); without torch, pointer arguments must be "
                   "metal_native.MetalBuffer objects",
                   i, Py_TYPE(arg)->tp_name);
      return false;
    }
  }
  return true;
}

static void encodeDispatch(MetalKernelObject *self, const LaunchGeometry &geom,
                           const std::vector<ArgInfo> &argInfos) {
  @autoreleasepool {
    get_device();
    reapInflight();
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self->pso];
    if (geom.tgmem > 0)
      [enc setThreadgroupMemoryLength:geom.tgmem atIndex:0];
    for (size_t i = 0; i < argInfos.size(); i++) {
      const ArgInfo &info = argInfos[i];
      switch (info.kind) {
      case ArgInfo::BUFFER:
        [enc setBuffer:info.buf offset:0 atIndex:i];
        break;
      case ArgInfo::INT:
        [enc setBytes:&info.intVal length:sizeof(int64_t) atIndex:i];
        break;
      case ArgInfo::FLOAT:
        [enc setBytes:&info.floatVal length:sizeof(float) atIndex:i];
        break;
      case ArgInfo::BYTES:
        [enc setBytes:info.bytesPtr length:(NSUInteger)info.bytesLen atIndex:i];
        break;
      }
    }
    MTLSize threadgroups =
        MTLSizeMake(geom.tx / geom.gx, geom.ty / geom.gy, geom.tz / geom.gz);
    MTLSize threadsPerGroup = MTLSizeMake(geom.gx, geom.gy, geom.gz);
    [enc dispatchThreadgroups:threadgroups
        threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    Inflight held{cb, {}};
    for (const ArgInfo &info : argInfos)
      if (info.kind == ArgInfo::BUFFER)
        held.held.push_back(Py_NewRef(info.obj));
    g_inflight.push_back(std::move(held));
    [cb commit];
    g_last = cb;
  }
}

static PyObject *MetalKernel_call(MetalKernelObject *self, PyObject *args,
                                  PyObject *kwargs) {
  LaunchGeometry geom;
  if (!readLaunchGeometry(kwargs, &geom))
    return NULL;
  std::vector<ArgInfo> argInfos;
  if (!packArguments(args, &argInfos))
    return NULL;
  encodeDispatch(self, geom, argInfos);
  Py_RETURN_NONE;
}

static PyObject *MetalKernel_get_max_threads(MetalKernelObject *self,
                                             void *closure) {
  return PyLong_FromUnsignedLongLong(self->maxThreads);
}

static PyGetSetDef MetalKernel_getset[] = {{"max_total_threads_per_threadgroup",
                                            (getter)MetalKernel_get_max_threads,
                                            NULL, NULL, NULL},
                                           {NULL}};

static PyTypeObject MetalKernelType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "metal_native.MetalKernel",
    .tp_basicsize = sizeof(MetalKernelObject),
    .tp_dealloc = (destructor)MetalKernel_dealloc,
    .tp_call = (ternaryfunc)MetalKernel_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = MetalKernel_getset,
};

// ── MetalLibrary ─────────────────────────────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLLibrary> library;
} MetalLibraryObject;

static void MetalLibrary_dealloc(MetalLibraryObject *self) {
  self->library = nil;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MetalLibrary_get_function(MetalLibraryObject *self,
                                           PyObject *args) {
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;
  @autoreleasepool {
    id<MTLFunction> fn = [self->library
        newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
      PyErr_Format(PyExc_KeyError, "Function '%s' not found", name);
      return NULL;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
        [get_device() newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
      NSMutableString *full = [NSMutableString string];
      appendErrorDetail(full, error);
      PyErr_Format(PyExc_RuntimeError, "PSO creation failed: %s",
                   [full UTF8String]);
      return NULL;
    }
    MetalKernelObject *kernel =
        (MetalKernelObject *)MetalKernelType.tp_alloc(&MetalKernelType, 0);
    if (!kernel)
      return NULL;
    kernel->pso = pso;
    kernel->maxThreads = [pso maxTotalThreadsPerThreadgroup];
    kernel->name = PyUnicode_FromString(name);
    return (PyObject *)kernel;
  }
}

static PyObject *MetalLibrary_get_function_names(MetalLibraryObject *self,
                                                 void *closure) {
  NSArray<NSString *> *names = [self->library functionNames];
  PyObject *list = PyList_New([names count]);
  for (NSUInteger i = 0; i < [names count]; i++)
    PyList_SetItem(list, i, PyUnicode_FromString([names[i] UTF8String]));
  return list;
}

static PyMethodDef MetalLibrary_methods[] = {
    {"get_function", (PyCFunction)MetalLibrary_get_function, METH_VARARGS,
     NULL},
    {NULL}};

static PyGetSetDef MetalLibrary_getset[] = {
    {"function_names", (getter)MetalLibrary_get_function_names, NULL, NULL,
     NULL},
    {NULL}};

static PyTypeObject MetalLibraryType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "metal_native.MetalLibrary",
    .tp_basicsize = sizeof(MetalLibraryObject),
    .tp_dealloc = (destructor)MetalLibrary_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = MetalLibrary_methods,
    .tp_getset = MetalLibrary_getset,
};

// ── Module functions ─────────────────────────────────────────────────────

static PyObject *py_load_metallib(PyObject *self, PyObject *args) {
  Py_buffer buf;
  if (!PyArg_ParseTuple(args, "y*", &buf))
    return NULL;
  @autoreleasepool {
    dispatch_data_t dd = dispatch_data_create(buf.buf, buf.len, nil,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    PyBuffer_Release(&buf);
    NSError *error = nil;
    id<MTLLibrary> lib = [get_device() newLibraryWithData:dd error:&error];
    if (!lib) {
      PyErr_Format(PyExc_RuntimeError, "Failed to load metallib: %s",
                   [[error localizedDescription] UTF8String]);
      return NULL;
    }
    MetalLibraryObject *obj =
        (MetalLibraryObject *)MetalLibraryType.tp_alloc(&MetalLibraryType, 0);
    if (!obj)
      return NULL;
    obj->library = lib;
    return (PyObject *)obj;
  }
}

static PyObject *py_compile_source(PyObject *self, PyObject *args) {
  const char *src, *math_mode;
  if (!PyArg_ParseTuple(args, "ss", &src, &math_mode))
    return NULL;
  @autoreleasepool {
    MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
    if (strcmp(math_mode, "safe") == 0) {
      if ([opt respondsToSelector:@selector(setMathMode:)]) {
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
      } else {
        opt.fastMathEnabled = NO;
      }
    }
    NSError *error = nil;
    id<MTLLibrary> lib =
        [get_device() newLibraryWithSource:[NSString stringWithUTF8String:src]
                                   options:opt
                                     error:&error];
    if (!lib) {
      NSMutableString *full = [NSMutableString string];
      appendErrorDetail(full, error);
      PyErr_Format(PyExc_RuntimeError, "MSL compile failed: %s",
                   [full UTF8String]);
      return NULL;
    }
    dispatch_data_t contents =
        [(id<MTLLibraryDataContents>)lib libraryDataContents];
    if (!contents) {
      PyErr_SetString(PyExc_RuntimeError, "MTLLibrary has no data contents");
      return NULL;
    }
    NSData *data = (NSData *)contents;
    if (![data length]) {
      PyErr_SetString(PyExc_RuntimeError, "metallib contents are empty");
      return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)[data bytes], [data length]);
  }
}

static PyObject *py_get_device_name(PyObject *self, PyObject *Py_UNUSED(args)) {
  return PyUnicode_FromString([[get_device() name] UTF8String]);
}

static PyObject *py_is_available(PyObject *self, PyObject *Py_UNUSED(args)) {
  return PyBool_FromLong(MTLCreateSystemDefaultDevice() != nil);
}

static PyObject *py_synchronize(PyObject *self, PyObject *Py_UNUSED(args)) {
  if (!sync_gpu())
    return NULL;
  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"alloc", py_alloc, METH_VARARGS, NULL},
    {"wrap", py_wrap, METH_VARARGS, NULL},
    {"load_metallib", py_load_metallib, METH_VARARGS, NULL},
    {"compile_source", py_compile_source, METH_VARARGS, NULL},
    {"get_device_name", py_get_device_name, METH_NOARGS, NULL},
    {"is_available", py_is_available, METH_NOARGS, NULL},
    {"synchronize", py_synchronize, METH_NOARGS, NULL},
    {NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "metal_native",
    "Triton Metal runtime without torch: own queue, MetalBuffer arguments", -1,
    module_methods};

PyMODINIT_FUNC PyInit_metal_native(void) {
  PyObject *m = PyModule_Create(&module_def);
  if (!m)
    return NULL;
  if (PyType_Ready(&MetalBufferType) < 0 ||
      PyType_Ready(&MetalKernelType) < 0 || PyType_Ready(&MetalLibraryType) < 0)
    return NULL;
  Py_INCREF(&MetalBufferType);
  Py_INCREF(&MetalKernelType);
  Py_INCREF(&MetalLibraryType);
  PyModule_AddObject(m, "MetalBuffer", (PyObject *)&MetalBufferType);
  PyModule_AddObject(m, "MetalKernel", (PyObject *)&MetalKernelType);
  PyModule_AddObject(m, "MetalLibrary", (PyObject *)&MetalLibraryType);
  return m;
}
