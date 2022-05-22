#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "ray_casting.hpp"

using namespace rc;

/* General template for iterating a numpy array */
#define PYARRAY_ITER_BEGIN(arr, err_msg, err_ret) {                     \
    NpyIter *_iter = NpyIter_New(                                        \
        arr,                                                            \
        NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,  \
        NPY_CORDER, NPY_NO_CASTING,                                     \
        NULL                                                            \
    );                                                                  \
    if (_iter == NULL) {                                                 \
        PyErr_SetString(PyExc_ValueError, err_msg);                     \
        return err_ret;                                                 \
    }                                                                   \
    NpyIter_IterNextFunc *_iternext = NpyIter_GetIterNext(_iter, NULL);   \
    if (_iternext == NULL) {                                             \
        NpyIter_Deallocate(_iter);                                       \
        PyErr_SetString(PyExc_ValueError, err_msg);                     \
        return err_ret;                                                 \
    }                                                                   \
    char **_dataptr = NpyIter_GetDataPtrArray(_iter);                     \
    npy_intp *_strideptr = NpyIter_GetInnerStrideArray(_iter);            \
    npy_intp *_innersizeptr = NpyIter_GetInnerLoopSizePtr(_iter);         \
    do {                                                                \
        char *data = *_dataptr;                                          \
        npy_intp _stride = *_strideptr;                                   \
        npy_intp _count = *_innersizeptr;                                 \
        while (_count--) {
#define PYARRAY_ITER_END                                                \
            data += _stride;                                             \
        }                                                               \
    } while(_iternext(_iter));                                            \
    NpyIter_Deallocate(_iter);                                           \
}


Geometry **parse_disks(PyObject *disks_obj, long int *num_disks) {
    PyArrayObject *disks_arr = (PyArrayObject *)PyArray_FromAny(disks_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    disks_arr = (PyArrayObject *)PyArray_Cast(disks_arr, NPY_FLOAT32);
    if (PyArray_NDIM(disks_arr) != 2 || PyArray_SHAPE(disks_arr)[1] != 7) {
        PyErr_SetString(PyExc_ValueError, "disks must be a (N, 7) array.");
        return NULL;
    }
    *num_disks = PyArray_SHAPE(disks_arr)[0];
    Geometry **disks = new Geometry *[*num_disks];
    float buf[7];
    long int idx = 0, count = 0;
    PYARRAY_ITER_BEGIN(disks_arr, "invalid disks array", NULL)
        buf[count++] = *(float *)data;
        if (count >= 7) {
            disks[idx] = new Disk(
                idx, glm::vec3(buf[0], buf[1], buf[2]), 
                glm::vec3(buf[3], buf[4], buf[5]), buf[6]
            );
            ++idx; count = 0;
        }
    PYARRAY_ITER_END
    return disks;
}

Ray *parse_rays(PyObject *rays_obj, long int *num_rays) {
    PyArrayObject *rays_arr = (PyArrayObject *)PyArray_FromAny(rays_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    rays_arr = (PyArrayObject *)PyArray_Cast(rays_arr, NPY_FLOAT32);
    if (PyArray_NDIM(rays_arr) != 2 || PyArray_SHAPE(rays_arr)[1] != 7) {
        PyErr_SetString(PyExc_ValueError, "rays must be a (M, 7) array.");
        return NULL;
    }
    *num_rays = PyArray_SHAPE(rays_arr)[0];
    Ray *rays = new Ray[*num_rays];
    float buf[7];
    long int idx = 0, count = 0;
    PYARRAY_ITER_BEGIN(rays_arr, "invalid rays array", NULL)
        buf[count++] = *(float *)data;
        if (count >= 7) {
            rays[idx] = {
                glm::vec3(buf[0], buf[1], buf[2]),
                glm::vec3(buf[3], buf[4], buf[5]), buf[6]
            };
            ++idx; count = 0;
        }
    PYARRAY_ITER_END
    return rays;
}


const char *PyDoc_CastRaysToDisks = ""
    "Performs ray casting without any acceleration data structure."
    ""
    "Args:"
    "    rays (np.ndarray): a (M, 7) dimensional array representing M rays"
    "        encoded by their origins, directions, and maximum range:"
    "        (o_x, o_y, o_z, d_x, d_y, d_z, range)"
    "    disks (np.ndarray): a (N, 7) dimensional array representing N disks"
    "        encoded by their position, normals, and sizes:"
    "        (x, y, z, n_x, n_y, n_z, size)"
    ""
    "Returns:"
    "    np.ndarray (M,): disk ids, np.nan if the ray does not hit"
    "    np.ndarray (M,): distance to the hit, np.inf if the ray does not hit"
;

static PyObject *
PyMethod_CastRaysToDisksNaive(PyObject *self, PyObject *args) {
    /* Parse arguments */
    PyObject *rays_obj, *disks_obj;
    if (!PyArg_ParseTuple(args, "OO", &rays_obj, &disks_obj)) return NULL;

    /* Convert and parse rays */
    long int num_rays;
    Ray *rays = parse_rays(rays_obj, &num_rays);
    if (rays == NULL) return NULL;

    /* Convert and parse disks */
    long int num_disks;
    Geometry **disks = parse_disks(disks_obj, &num_disks);
    if (disks == NULL) return NULL;

    Hit *result = new Hit[num_rays];
    cast_rays_naive(rays, num_rays, disks, num_disks, result);

    float *t = new float[num_rays];
    for (long int i = 0; i < num_rays; ++i)
        t[i] = result[i].t;
    npy_intp shape[1] = {num_rays};
    PyObject *t_arr = PyArray_SimpleNewFromData(1, shape, NPY_FLOAT32, t);

    /* Clean up */
    delete disks;
    delete rays;
    delete result;

    return t_arr;
}

static PyObject *
PyMethod_CastRaysToDisks(PyObject *self, PyObject *args) {
    /* Parse arguments */
    PyObject *rays_obj, *disks_obj;
    if (!PyArg_ParseTuple(args, "OO", &rays_obj, &disks_obj)) return NULL;

    /* Convert and parse rays */
    long int num_rays;
    Ray *rays = parse_rays(rays_obj, &num_rays);
    if (rays == NULL) return NULL;

    // PySys_WriteStdout("Num rays: %ld\n", num_rays);

    /* Convert and parse disks */
    long int num_disks;
    Geometry **disks = parse_disks(disks_obj, &num_disks);
    if (disks == NULL) return NULL;

    // PySys_WriteStdout("Num disks: %ld\n", num_disks);

    Hit *result = new Hit[num_rays];
    cast_rays_with_kdtree(rays, num_rays, disks, num_disks, result);

    float *t = new float[num_rays];
    for (long int i = 0; i < num_rays; ++i)
        t[i] = result[i].t;
    npy_intp shape[1] = {num_rays};
    PyObject *t_arr = PyArray_SimpleNewFromData(1, shape, NPY_FLOAT32, t);

    /* Clean up */
    for (long int i = 0; i < num_disks; ++i)
        delete disks[i];
    delete disks;
    delete rays;
    delete result;

    return t_arr;
}


static PyMethodDef method_defs[] = {
    {"cast_rays_to_disks_naive", PyMethod_CastRaysToDisksNaive, METH_VARARGS, PyDoc_CastRaysToDisks},
    {"cast_rays_to_disks", PyMethod_CastRaysToDisks, METH_VARARGS, PyDoc_CastRaysToDisks},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "ray_casting", NULL, -1, method_defs
};

PyMODINIT_FUNC
PyInit_ray_casting(void)
{
    PyObject *m = PyModule_Create(&module_def);
    import_array();
    return m;
}
