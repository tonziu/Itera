#include <cassert>
#include <cstdlib>

namespace math
{
    // Itera matrix model.
    typedef struct
    {
        int rows;
        int cols;
        double *data = nullptr;
    } Matrix;

    // Allocates memory for an uninitialized matrix.
    void matrix_alloc(Matrix &matrix, int rows, int cols)
    {
        assert(matrix.data == nullptr);
        assert(rows > 0);
        assert(cols > 0);

        matrix.data = (double *)malloc(rows * cols * sizeof(double));
        assert(matrix.data != nullptr);

        matrix.rows = rows;
        matrix.cols = cols;
    }

    // Fills a valid matrix with a given value.
    // Use matrix_alloc first to allocate memory.
    void matrix_fill(Matrix &matrix, double value)
    {
        assert(matrix.data);
        assert(matrix.rows > 0);
        assert(matrix.cols > 0);

        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            matrix.data[i] = value;
        }

        return;
    }

    // Frees matrix data.
    void matrix_free(Matrix &matrix)
    {
        assert(matrix.data);
        assert(matrix.rows > 0);
        assert(matrix.cols > 0);    

        free(matrix.data);
        matrix.data = nullptr;
        matrix.rows = 0;
        matrix.cols = 0;
    }
}
