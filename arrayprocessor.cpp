#include "arrayprocessor.h"
#include <cmath>

void process_array_inplace(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = sin(data[i]) * cos(data[i]);
    }
}

double* create_processed_array(double* input, int size) {
    double* output = new double[size];
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * input[i] + 2.0 * input[i] + 1.0;
    }
    return output;
}