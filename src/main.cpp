#include <iostream>

#include <math/matrix.h>

using namespace math;

int main(void)
{
    math::Matrix mat;

    // Allocate a 2x3 matrix
    math::matrix_alloc(mat, 2, 3);

    // Initialize the matrix with known values
    mat.data[0] = 1.0;
    mat.data[1] = -2.0;
    mat.data[2] = 0.5;
    mat.data[3] = 3.0;
    mat.data[4] = -1.0;
    mat.data[5] = 2.0;

    std::cout << "Original matrix:" << std::endl;
    matrix_print(mat);

    // Apply ReLU
    math::matrix_relu_in_place(mat);
    std::cout << "After ReLU:" << std::endl;
    matrix_print(mat);

    // Reinitialize matrix with original values for demonstration
    mat.data[0] = 1.0;
    mat.data[1] = -2.0;
    mat.data[2] = 0.5;
    mat.data[3] = 3.0;
    mat.data[4] = -1.0;
    mat.data[5] = 2.0;

    // Apply Sigmoid
    math::matrix_sigmoid_in_place(mat);
    std::cout << "After Sigmoid:" << std::endl;
    matrix_print(mat);

    // Reinitialize matrix with original values for demonstration
    mat.data[0] = 1.0;
    mat.data[1] = -2.0;
    mat.data[2] = 0.5;
    mat.data[3] = 3.0;
    mat.data[4] = -1.0;
    mat.data[5] = 2.0;

    // Apply Tanh
    math::matrix_tanh_in_place(mat);
    std::cout << "After Tanh:" << std::endl;
    matrix_print(mat);

    // Reinitialize matrix with original values for demonstration
    mat.data[0] = 1.0;
    mat.data[1] = -2.0;
    mat.data[2] = 0.5;
    mat.data[3] = 3.0;
    mat.data[4] = -1.0;
    mat.data[5] = 2.0;

    // Apply Softmax
    math::matrix_softmax_in_place(mat);
    std::cout << "After Softmax:" << std::endl;
    matrix_print(mat);

    // Free allocated memory
    math::matrix_free(mat);

    return 0;
}