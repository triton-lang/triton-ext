// metal_utils — Triton Apple backend Metal runtime
//
// JIT-compiled at first use. Links against user's installed libtorch
// to access getMTLBufferStorage() for zero-copy MPS tensor dispatch.
//
// Works with any PyTorch version that has MPS support (2.0+).
// No custom PyTorch wheel needed.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// From PyTorch: ATen/native/mps/OperationUtils.h
// getMTLBufferStorage = __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data())
// We link against libtorch to get THPVariable_Unpack.
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/Tensor.h>
#include <ATen/mps/MPSStream.h>

static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& t) {
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

// Use PyTorch's MPS device — same device that owns the tensor buffers.
static id<MTLDevice> get_device(void) {
    return at::mps::getCurrentMPSStream()->device();
}

// ── MetalKernel — callable PSO wrapper ───────────────────────────────────

typedef struct {
    PyObject_HEAD
    id<MTLComputePipelineState> pso;
    NSUInteger maxThreads;
} MetalKernelObject;

static void MetalKernel_dealloc(MetalKernelObject *self) {
    self->pso = nil;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MetalKernel_call(MetalKernelObject *self,
                                   PyObject *args, PyObject *kwargs) {
    PyObject *threads_obj = NULL, *group_obj = NULL;
    if (kwargs) {
        threads_obj = PyDict_GetItemString(kwargs, "threads");
        group_obj = PyDict_GetItemString(kwargs, "group_size");
    }
    if (!threads_obj || !group_obj) {
        PyErr_SetString(PyExc_ValueError, "threads and group_size required");
        return NULL;
    }

    long tx = PyLong_AsLong(PyList_GetItem(threads_obj, 0));
    long ty = PyLong_AsLong(PyList_GetItem(threads_obj, 1));
    long tz = PyLong_AsLong(PyList_GetItem(threads_obj, 2));
    long gx = PyLong_AsLong(PyList_GetItem(group_obj, 0));
    long gy = PyLong_AsLong(PyList_GetItem(group_obj, 1));
    long gz = PyLong_AsLong(PyList_GetItem(group_obj, 2));

    Py_ssize_t nargs = PyTuple_Size(args);

    // Pre-extract tensor buffers and scalar values before entering the
    // Metal dispatch block (Python API calls not safe inside dispatch_sync).
    struct ArgInfo {
        enum Kind { TENSOR, INT, FLOAT } kind;
        id<MTLBuffer> buf;
        NSUInteger offset;
        int64_t intVal;
        float floatVal;
    };
    std::vector<ArgInfo> argInfos(nargs);
    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject *arg = PyTuple_GetItem(args, i);
        if (THPVariable_Check(arg)) {
            at::Tensor t = THPVariable_Unpack(arg);
            if (!t.is_mps()) {
                PyErr_Format(PyExc_RuntimeError,
                    "Arg %zd: tensor must be on MPS device, got %s",
                    i, t.device().str().c_str());
                return NULL;
            }
            argInfos[i].kind = ArgInfo::TENSOR;
            argInfos[i].buf = getMTLBufferStorage(t);
            argInfos[i].offset = t.storage_offset() * t.element_size();
        } else if (PyLong_Check(arg)) {
            argInfos[i].kind = ArgInfo::INT;
            argInfos[i].intVal = PyLong_AsLongLong(arg);
        } else if (PyFloat_Check(arg)) {
            argInfos[i].kind = ArgInfo::FLOAT;
            argInfos[i].floatVal = (float)PyFloat_AsDouble(arg);
        } else {
            PyErr_Format(PyExc_TypeError,
                "Arg %zd: expected tensor, int, or float", i);
            return NULL;
        }
    }

    // Use PyTorch's MPS stream directly. Synchronize first to ensure
    // any pending MPS graph operations are flushed before we encode.
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        [enc setComputePipelineState:self->pso];

        for (Py_ssize_t i = 0; i < nargs; i++) {
            auto &info = argInfos[i];
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
            }
        }

        MTLSize threadgroups = MTLSizeMake(tx / gx, ty / gy, tz / gz);
        MTLSize threadsPerGroup = MTLSizeMake(gx, gy, gz);
        [enc dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerGroup];

        stream->endKernelCoalescing();
    }

    Py_RETURN_NONE;
}

static PyObject *MetalKernel_get_max_threads(MetalKernelObject *self, void *closure) {
    return PyLong_FromUnsignedLongLong(self->maxThreads);
}

static PyGetSetDef MetalKernel_getset[] = {
    {"max_total_threads_per_threadgroup",
     (getter)MetalKernel_get_max_threads, NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject MetalKernelType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "metal_utils.MetalKernel",
    .tp_basicsize = sizeof(MetalKernelObject),
    .tp_dealloc = (destructor)MetalKernel_dealloc,
    .tp_call = (ternaryfunc)MetalKernel_call,
    .tp_getset = MetalKernel_getset,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// ── MetalLibrary — metallib container ────────────────────────────────────

typedef struct {
    PyObject_HEAD
    id<MTLLibrary> library;
} MetalLibraryObject;

static void MetalLibrary_dealloc(MetalLibraryObject *self) {
    self->library = nil;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MetalLibrary_get_function(MetalLibraryObject *self,
                                            PyObject *args) {
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) return NULL;

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
        PyErr_Format(PyExc_RuntimeError, "PSO creation failed: %s",
                     [[error localizedDescription] UTF8String]);
        return NULL;
    }

    MetalKernelObject *kernel = PyObject_New(MetalKernelObject, &MetalKernelType);
    kernel->pso = pso;
    kernel->maxThreads = [pso maxTotalThreadsPerThreadgroup];
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
    {"get_function", (PyCFunction)MetalLibrary_get_function, METH_VARARGS, NULL},
    {NULL}
};

static PyGetSetDef MetalLibrary_getset[] = {
    {"function_names", (getter)MetalLibrary_get_function_names, NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject MetalLibraryType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "metal_utils.MetalLibrary",
    .tp_basicsize = sizeof(MetalLibraryObject),
    .tp_dealloc = (destructor)MetalLibrary_dealloc,
    .tp_methods = MetalLibrary_methods,
    .tp_getset = MetalLibrary_getset,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// ── Module functions ─────────────────────────────────────────────────────

static PyObject *py_load_metallib(PyObject *self, PyObject *args) {
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf)) return NULL;

    @autoreleasepool {
        dispatch_data_t dd = dispatch_data_create(
            buf.buf, buf.len, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        PyBuffer_Release(&buf);

        NSError *error = nil;
        id<MTLLibrary> lib = [get_device() newLibraryWithData:dd error:&error];
        if (!lib) {
            PyErr_Format(PyExc_RuntimeError, "Failed to load metallib: %s",
                         [[error localizedDescription] UTF8String]);
            return NULL;
        }

        MetalLibraryObject *obj = PyObject_New(MetalLibraryObject, &MetalLibraryType);
        obj->library = lib;
        return (PyObject *)obj;
    }
}

static PyObject *py_get_device_name(PyObject *self, PyObject *Py_UNUSED(args)) {
    return PyUnicode_FromString([[get_device() name] UTF8String]);
}

static PyObject *py_is_available(PyObject *self, PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(MTLCreateSystemDefaultDevice() != nil);
}

static PyMethodDef module_methods[] = {
    {"load_metallib", py_load_metallib, METH_VARARGS, NULL},
    {"get_device_name", py_get_device_name, METH_NOARGS, NULL},
    {"is_available", py_is_available, METH_NOARGS, NULL},
    {NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "metal_utils",
    "Triton Metal runtime — zero-copy MPS dispatch via libtorch", -1,
    module_methods
};

PyMODINIT_FUNC PyInit_metal_utils(void) {
    PyObject *m = PyModule_Create(&module_def);
    if (!m) return NULL;
    if (PyType_Ready(&MetalKernelType) < 0) return NULL;
    if (PyType_Ready(&MetalLibraryType) < 0) return NULL;
    Py_INCREF(&MetalKernelType);
    Py_INCREF(&MetalLibraryType);
    PyModule_AddObject(m, "MetalKernel", (PyObject *)&MetalKernelType);
    PyModule_AddObject(m, "MetalLibrary", (PyObject *)&MetalLibraryType);
    return m;
}
