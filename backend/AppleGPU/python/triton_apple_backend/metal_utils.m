// metal_utils - Triton Apple backend Metal runtime. Links against the user's
// libtorch for getMTLBufferStorage() (zero-copy MPS tensor dispatch).
// Requires PyTorch 2.0+ with MPS.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Undeclared in the SDK headers, so the call site would otherwise infer `id`.
@protocol MTLLibraryDataContents
- (dispatch_data_t)libraryDataContents;
@end

// getMTLBufferStorage mirrors PyTorch's ATen/native/mps/OperationUtils.h.
#include <ATen/Tensor.h>
#include <ATen/mps/MPSStream.h>
#include <torch/csrc/autograd/python_variable.h>

static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase &t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

// Use PyTorch's MPS device - same device that owns the tensor buffers.
static id<MTLDevice> get_device(void) {
  return at::mps::getCurrentMPSStream()->device();
}

// ── per-kernel GPU timing ────────────────────────────────────────────────
//
// A compute encoder is one stage and the fast path shares one encoder across
// back-to-back dispatches, so profiling ends the shared encoder and gives the
// dispatch its own, which serialises the kernels. Opt-in, off by default.
static struct {
  bool on;
  id<MTLCounterSampleBuffer> sampleBuf;
  NSMutableArray<NSString *> *names;
  NSMutableArray<NSNumber *> *nanos;
} g_prof = {false, nil, nil, nil};

static id<MTLCounterSet> timestampCounterSet(id<MTLDevice> dev) {
  for (id<MTLCounterSet> cs in dev.counterSets)
    if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp])
      return cs;
  return nil;
}

// ── MetalKernel - callable PSO wrapper ───────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLComputePipelineState> pso;
  NSUInteger maxThreads;
  PyObject *name; // for the profile's rows
} MetalKernelObject;

static void MetalKernel_dealloc(MetalKernelObject *self) {
  self->pso = nil;
  Py_CLEAR(self->name);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// One kernel argument, read out of Python before the dispatch block: Python
// API calls are not safe inside dispatch_sync.
struct ArgInfo {
  enum Kind { TENSOR, INT, FLOAT, BYTES } kind;
  id<MTLBuffer> buf;
  NSUInteger offset;
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

// False with the Python error set.
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

  // Dynamic threadgroup memory: when nonzero the kernel declares a trailing
  // addrspace(3) param; bind its byte length at TG location-index 0 (a separate
  // index space from device setBuffer, so no clash with the buffer args).
  out->tgmem = tgmem_obj ? PyLong_AsLong(tgmem_obj) : 0;

  out->tx = PyLong_AsLong(PyList_GetItem(threads_obj, 0));
  out->ty = PyLong_AsLong(PyList_GetItem(threads_obj, 1));
  out->tz = PyLong_AsLong(PyList_GetItem(threads_obj, 2));
  out->gx = PyLong_AsLong(PyList_GetItem(group_obj, 0));
  out->gy = PyLong_AsLong(PyList_GetItem(group_obj, 1));
  out->gz = PyLong_AsLong(PyList_GetItem(group_obj, 2));
  return true;
}

// False with the Python error set.
static bool packArguments(PyObject *args, std::vector<ArgInfo> *out) {
  const Py_ssize_t nargs = PyTuple_Size(args);
  out->resize(nargs);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject *arg = PyTuple_GetItem(args, i);
    ArgInfo &info = (*out)[i];
    if (THPVariable_Check(arg)) {
      at::Tensor t = THPVariable_Unpack(arg);
      // Guard before t.device(): a storage-less tensor (ABI-skewed libtorch)
      // makes t.device() abort with an uncatchable c10::Error.
      if (!t.defined() || !t.has_storage()) {
        PyErr_Format(PyExc_RuntimeError,
                     "Arg %zd: tensor is undefined or has no storage "
                     "(rebuild metal_utils against the current libtorch?)",
                     i);
        return false;
      }
      if (!t.is_mps()) {
        PyErr_Format(PyExc_RuntimeError,
                     "Arg %zd: tensor must be on MPS device, got %s", i,
                     t.device().str().c_str());
        return false;
      }
      info.kind = ArgInfo::TENSOR;
      info.buf = getMTLBufferStorage(t);
      const uintptr_t base = (uintptr_t)[info.buf contents];
      const uintptr_t data = (uintptr_t)t.data_ptr();
      info.offset = data >= base && data - base < [info.buf length]
                        ? (NSUInteger)(data - base)
                        : t.storage_offset() * t.element_size();
    } else if (PyBytes_Check(arg)) {
      // Packed scalar blob, bound inline via setBytes. The args tuple keeps
      // the object alive across the dispatch_sync below.
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
                   "Arg %zd: expected tensor, int, float, or bytes", i);
      return false;
    }
  }
  return true;
}

static void bindArguments(id<MTLComputeCommandEncoder> enc,
                          const std::vector<ArgInfo> &argInfos) {
  for (size_t i = 0; i < argInfos.size(); i++) {
    const ArgInfo &info = argInfos[i];
    switch (info.kind) {
    case ArgInfo::TENSOR:
      [enc setBuffer:info.buf offset:info.offset atIndex:i];
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
}

static void encodeDispatch(MetalKernelObject *self, const LaunchGeometry &geom,
                           const std::vector<ArgInfo> &argInfos) {
  // Dispatch on stream->queue() (serial) to serialize with other MPS ops.
  // Don't call endKernelCoalescing(): reusing torch's cached encoder
  // coalesces back-to-back dispatches and RAW deps are still honored.
  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        const NSUInteger sampleIdx = g_prof.names ? g_prof.names.count * 2 : 0;
        const bool sampling =
            g_prof.on && g_prof.sampleBuf && sampleIdx + 1 < 4096;
        // Samples resolve at stage (encoder) boundaries wherever in the
        // encoder they are requested, so a dispatch sharing an encoder reads
        // the whole encoder's interval. Isolate it.
        if (sampling)
          stream->endKernelCoalescing();

        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

        if (sampling)
          [enc sampleCountersInBuffer:g_prof.sampleBuf
                        atSampleIndex:sampleIdx
                          withBarrier:NO];

        [enc setComputePipelineState:self->pso];

        if (geom.tgmem > 0)
          [enc setThreadgroupMemoryLength:geom.tgmem atIndex:0];

        bindArguments(enc, argInfos);

        MTLSize threadgroups = MTLSizeMake(geom.tx / geom.gx, geom.ty / geom.gy,
                                           geom.tz / geom.gz);
        MTLSize threadsPerGroup = MTLSizeMake(geom.gx, geom.gy, geom.gz);
        [enc dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerGroup];

        // Close the sample, then end the encoder so its stage-end boundary,
        // which is what the sample reads, lands right after this dispatch.
        if (sampling) {
          [enc sampleCountersInBuffer:g_prof.sampleBuf
                        atSampleIndex:sampleIdx + 1
                          withBarrier:NO];
          stream->endKernelCoalescing();
          const char *nm = self->name ? PyUnicode_AsUTF8(self->name) : NULL;
          [g_prof.names addObject:@(nm ? nm : "triton_kernel")];
        }
      }
    });
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
        "metal_utils.MetalKernel",
    .tp_basicsize = sizeof(MetalKernelObject),
    .tp_dealloc = (destructor)MetalKernel_dealloc,
    .tp_call = (ternaryfunc)MetalKernel_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = MetalKernel_getset,
};

// ── MetalLibrary - metallib container ────────────────────────────────────

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

  NSString *fnName = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [self->library newFunctionWithName:fnName];
  if (!fn) {
    PyErr_Format(PyExc_KeyError, "Function '%s' not found", name);
    return NULL;
  }

  NSError *error = nil;
  id<MTLComputePipelineState> pso =
      [get_device() newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) {
    // The PSO compiler often leaves localizedDescription useless and puts the
    // real diagnostic in userInfo or the underlying error.
    NSMutableString *full = [NSMutableString string];
    [full appendFormat:@"%@", [error localizedDescription]];
    NSDictionary *info = [error userInfo];
    if (info && [info count])
      [full appendFormat:@" | userInfo=%@", info];
    NSError *under = [[error userInfo] objectForKey:NSUnderlyingErrorKey];
    if (under)
      [full appendFormat:@" | underlying=%@ (%@)", [under localizedDescription],
                         [under userInfo]];
    PyErr_Format(PyExc_RuntimeError, "PSO creation failed: %s",
                 [full UTF8String]);
    return NULL;
  }

  MetalKernelObject *kernel = PyObject_New(MetalKernelObject, &MetalKernelType);
  kernel->pso = pso;
  kernel->maxThreads = [pso maxTotalThreadsPerThreadgroup];
  kernel->name = PyUnicode_FromString(name);
  return (PyObject *)kernel;
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
        "metal_utils.MetalLibrary",
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
        PyObject_New(MetalLibraryObject, &MetalLibraryType);
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
    // Fast-math assumes no NaN/Inf and reassociates FP, so it miscompiles
    // kernels over Inf/NaN or relying on RTNE. Transcendental accuracy is
    // chosen per-op via the metal::precise:: namespace the emitter selects,
    // which holds even under MTLMathFloatingPointFunctionsFast; forcing
    // Precise here would apply to every op.
    if (strcmp(math_mode, "safe") == 0) {
      if ([opt respondsToSelector:@selector(setMathMode:)]) {
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
      } else {
        opt.fastMathEnabled = NO;
      }
    }

    NSString *source = [NSString stringWithUTF8String:src];
    NSError *error = nil;
    id<MTLLibrary> lib = [get_device() newLibraryWithSource:source
                                                    options:opt
                                                      error:&error];
    if (!lib) {
      NSMutableString *full = [NSMutableString string];
      [full appendFormat:@"%@", [error localizedDescription]];
      NSDictionary *info = [error userInfo];
      if (info && [info count])
        [full appendFormat:@" | userInfo=%@", info];
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

// Begin per-kernel GPU timing. Fails when the device has no timestamp
// counter set.
static PyObject *py_profile_start(PyObject *self, PyObject *Py_UNUSED(args)) {
  id<MTLDevice> dev = get_device();
  id<MTLCounterSet> ts = timestampCounterSet(dev);
  if (!ts) {
    PyErr_SetString(PyExc_RuntimeError,
                    "this device exposes no timestamp counter set");
    return NULL;
  }
  // Don't trust `supportsCounterSampling:AtStageBoundary`: some devices answer
  // YES and then assert inside `-[AGX...ComputeContext
  // sampleCountersInBuffer:...]`, killing the process. No error comes back.
  // Gate on dispatch-boundary support instead; a device without it also
  // cannot take the stage-boundary path.
  if (![dev
          supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "this device cannot sample counters around a dispatch: it reports no "
        "dispatch-boundary support and its stage-boundary answer is not "
        "honoured by the compute encoder. Use the command-buffer totals "
        "instead.");
    return NULL;
  }
  MTLCounterSampleBufferDescriptor *d =
      [[MTLCounterSampleBufferDescriptor alloc] init];
  d.counterSet = ts;
  d.storageMode = MTLStorageModeShared;
  d.sampleCount = 4096;
  NSError *err = nil;
  g_prof.sampleBuf = [dev newCounterSampleBufferWithDescriptor:d error:&err];
  if (!g_prof.sampleBuf) {
    PyErr_Format(PyExc_RuntimeError, "counter sample buffer: %s",
                 err.localizedDescription.UTF8String);
    return NULL;
  }
  g_prof.names = [NSMutableArray array];
  g_prof.nanos = [NSMutableArray array];
  g_prof.on = true;
  Py_RETURN_NONE;
}

// End timing and return [(kernel name, gpu nanoseconds), ...], one row per
// dispatch in the order they ran.
static PyObject *py_profile_stop(PyObject *self, PyObject *Py_UNUSED(args)) {
  g_prof.on = false;
  if (!g_prof.sampleBuf)
    return PyList_New(0);

  // The GPU must have finished writing the samples before they are read.
  at::mps::getCurrentMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);

  const NSUInteger n = g_prof.names.count;
  PyObject *out = PyList_New((Py_ssize_t)n);
  if (!out)
    return NULL;
  NSData *data = [g_prof.sampleBuf resolveCounterRange:NSMakeRange(0, n * 2)];
  const MTLCounterResultTimestamp *t =
      data ? (const MTLCounterResultTimestamp *)data.bytes : NULL;
  for (NSUInteger i = 0; i < n; ++i) {
    // A pair the GPU did not write reads as MTLCounterErrorValue, which
    // subtracts to an enormous duration. Report zero.
    unsigned long long ns = 0;
    if (t && t[i * 2].timestamp != MTLCounterErrorValue &&
        t[i * 2 + 1].timestamp != MTLCounterErrorValue &&
        t[i * 2 + 1].timestamp >= t[i * 2].timestamp)
      ns = t[i * 2 + 1].timestamp - t[i * 2].timestamp;
    PyList_SET_ITEM(out, (Py_ssize_t)i,
                    Py_BuildValue("(sK)", g_prof.names[i].UTF8String, ns));
  }
  g_prof.sampleBuf = nil;
  g_prof.names = nil;
  g_prof.nanos = nil;
  return out;
}

static PyObject *py_is_available(PyObject *self, PyObject *Py_UNUSED(args)) {
  return PyBool_FromLong(MTLCreateSystemDefaultDevice() != nil);
}

static PyMethodDef module_methods[] = {
    {"load_metallib", py_load_metallib, METH_VARARGS, NULL},
    {"compile_source", py_compile_source, METH_VARARGS, NULL},
    {"get_device_name", py_get_device_name, METH_NOARGS, NULL},
    {"is_available", py_is_available, METH_NOARGS, NULL},
    {"profile_start", py_profile_start, METH_NOARGS, NULL},
    {"profile_stop", py_profile_stop, METH_NOARGS, NULL},
    {NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "metal_utils",
    "Triton Metal runtime: zero-copy MPS dispatch via libtorch", -1,
    module_methods};

PyMODINIT_FUNC PyInit_metal_utils(void) {
  PyObject *m = PyModule_Create(&module_def);
  if (!m)
    return NULL;
  if (PyType_Ready(&MetalKernelType) < 0)
    return NULL;
  if (PyType_Ready(&MetalLibraryType) < 0)
    return NULL;
  Py_INCREF(&MetalKernelType);
  Py_INCREF(&MetalLibraryType);
  PyModule_AddObject(m, "MetalKernel", (PyObject *)&MetalKernelType);
  PyModule_AddObject(m, "MetalLibrary", (PyObject *)&MetalLibraryType);
  return m;
}
