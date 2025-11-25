#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "arrayprocessor.h"

static PyObject* process_inplace(PyObject* self, PyObject* args) {
    PyArrayObject *arr;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) {
        return NULL;
    }

    if (PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type double!");
        return NULL;
    }

    double* data = (double*)PyArray_DATA(arr);
    int size = PyArray_DIM(arr, 0);

    process_array_inplace(data, size);

    Py_RETURN_NONE;
}

static PyObject* create_processed(PyObject* self, PyObject* args) {
    PyArrayObject *input_arr;
    npy_intp dims[1];

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_arr)) {
        return NULL;
    }

    if (PyArray_TYPE(input_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type double");
        return NULL;
    }

    double* input_data = (double*)PyArray_DATA(input_arr);
    int size = PyArray_DIM(input_arr, 0);
    dims[0] = size;

    PyArrayObject* output_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_arr == NULL) {
        return NULL;
    }

    double* processed_data = create_processed_array(input_data, size);
    double* output_data = (double*)PyArray_DATA(output_arr);

    for (int i = 0; i < size; i++) {
        output_data[i] = processed_data[i];
    }

    delete[] processed_data;

    return PyArray_Return(output_arr);
}

static PyObject* matrix_multiply(PyObject* self, PyObject* args) {
    PyArrayObject *matrix1, *matrix2;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &matrix1, &PyArray_Type, &matrix2)) {
        return NULL;
    }

    if (PyArray_NDIM(matrix1) != 2 || PyArray_NDIM(matrix2) != 2) {
        PyErr_SetString(PyExc_ValueError, "Both arrays must be 2-dimensional");
        return NULL;
    }

    int rows1 = PyArray_DIM(matrix1, 0);
    int cols1 = PyArray_DIM(matrix1, 1);
    int rows2 = PyArray_DIM(matrix2, 0);
    int cols2 = PyArray_DIM(matrix2, 1);

    if (cols1 != rows2) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions don't match for multiplication");
        return NULL;
    }

    npy_intp result_dims[2] = {rows1, cols2};
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);
    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols1; k++) {
                double val1 = *(double*)PyArray_GETPTR2(matrix1, i, k);
                double val2 = *(double*)PyArray_GETPTR2(matrix2, k, j);
                sum += val1 * val2;
            }
            *(double*)PyArray_GETPTR2(result, i, j) = sum;
        }
    }

    return PyArray_Return(result);
}

static PyMethodDef ArrayProcessorMethods[] = {
    {"process_inplace", process_inplace, METH_VARARGS, "Process array in-place using sin(x)*cos(x) function"},
    {"create_processed", create_processed, METH_VARARGS, "Create new array with processed values using x^2 + 2x + 1"},
    {"matrix_multiply", matrix_multiply, METH_VARARGS, "Multiply two 2D matrices"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef arrayprocessormodule = {
    PyModuleDef_HEAD_INIT,
    "arrayprocessor",
    "Module for array processing demonstration",
    -1,
    ArrayProcessorMethods
};

PyMODINIT_FUNC PyInit_arrayprocessor(void) {
    import_array();
    return PyModule_Create(&arrayprocessormodule);
}