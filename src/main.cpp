#include <iostream>

#include <math/matrix.h>

using namespace math;

int main(void)
{
    std::cout << "Hello from Itera.\n";

    Matrix matrix;
    matrix_alloc(matrix, 2, 2);
    matrix_fill(matrix, 2.3);

    for (int i = 0; i < matrix.rows*matrix.cols; ++i)
    {
        std::cout << matrix.data[i] << "\n";
    }

    matrix_free(matrix);

    return 0;
}