#include <iostream>

#include <math/matrix.h>

using namespace math;

int main(void)
{
    // Create two matrices for multiplication
    math::Matrix matrix_a;
    math::Matrix matrix_b;
    math::Matrix matrix_c;
    // Allocate matrices
    
    math::matrix_alloc(matrix_a, 2, 3); // 2x3 matrix
    math::matrix_alloc(matrix_b, 3, 2); // 3x2 matrix
    math::matrix_alloc(matrix_c, 2, 2);

    // Fill matrix_a with values
    matrix_a.data[0] = 1.0;
    matrix_a.data[1] = 2.0;
    matrix_a.data[2] = 3.0;
    matrix_a.data[3] = 4.0;
    matrix_a.data[4] = 5.0;
    matrix_a.data[5] = 6.0;

    // Fill matrix_b with values
    matrix_b.data[0] = 7.0;
    matrix_b.data[1] = 8.0;
    matrix_b.data[2] = 9.0;
    matrix_b.data[3] = 10.0;
    matrix_b.data[4] = 11.0;
    matrix_b.data[5] = 12.0;

    // Expected result of matrix_a * matrix_b:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    // = [58, 64]
    //   [139, 154]

    // Perform in-place matrix multiplication
    math::matrix_prod(matrix_c, matrix_a, matrix_b);

    // Print the result
    std::cout << "Resultant matrix after multiplication:" << std::endl;
    for (int i = 0; i < matrix_c.rows; ++i)
    {
        for (int j = 0; j < matrix_c.cols; ++j)
        {
            std::cout << matrix_c.data[i * matrix_c.cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    math::matrix_free(matrix_a);
    math::matrix_free(matrix_b);
    math::matrix_free(matrix_c);

    return 0;
}