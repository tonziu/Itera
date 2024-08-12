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

        return;
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

        return;
    }

    // Adds the values of 'b' to 'a' if the matrices have matching sizes.
    void matrix_add_inplace(Matrix &a, Matrix &b)
    {
        assert(a.data != nullptr);
        assert(b.data != nullptr);
        assert(a.rows > 0);
        assert(a.cols > 0);
        assert(a.rows == b.rows);
        assert(a.cols == b.cols);

        for (int i = 0; i < a.rows * a.cols; ++i)
        {
            a.data[i] += b.data[i];
        }

        return;
    }

    // Copy the values of 'src' to 'dst' if the matrices have matching sizes.
    // The destination matrix needs to be allocated first using matrix_alloc.
    void matrix_copy(const Matrix &src, Matrix &dst)
    {
        assert(src.data != nullptr);
        assert(dst.data != nullptr);
        assert(src.rows > 0);
        assert(src.cols > 0);
        assert(src.rows == dst.rows);
        assert(src.cols == dst.cols);

        for (int i = 0; i < src.rows * src.cols; ++i)
        {
            dst.data[i] = src.data[i];
        }

        return;
    }

    // In place multiplication 'left * right' if matrices sizes fit correctly.
    void matrix_prod_in_place(Matrix &left, Matrix &right)
    {
        assert(left.data != nullptr);
        assert(right.data != nullptr);
        assert(left.rows > 0);
        assert(left.cols > 0);
        assert(left.cols == right.rows);
        assert(right.cols > 0);

        Matrix result;
        matrix_alloc(result, left.rows, right.cols);

        for (int i = 0; i < left.rows; ++i)
        {
            for (int j = 0; j < right.cols; ++j)
            {
                double accumulated = 0.0;
                for (int k = 0; k < left.cols; ++k)
                {
                    accumulated += left.data[i * left.cols + k] * right.data[k * right.cols + j];
                }
                result.data[i * right.cols + j] = accumulated;
            }
        }

        free(left.data);
        left.data = result.data;
        left.cols = right.cols;
        result.data = nullptr;
    }

    void matrix_add(Matrix &out, Matrix &a, Matrix &b)
    {
        assert(out.data != nullptr);
        assert(a.data != nullptr);
        assert(b.data != nullptr);
        assert(out.rows > 0);
        assert(out.cols > 0);
        assert(out.rows == a.rows && a.rows == b.rows);
        assert(out.cols == a.cols && a.cols == b.cols);

        for (int i = 0; i < out.rows * out.cols; ++i)
        {
            out.data[i] = a.data[i] + b.data[i];
        }

        return;
    }

    void matrix_prod(Matrix &out, Matrix &a, Matrix &b)
    {
        assert(out.data != nullptr);
        assert(a.data != nullptr);
        assert(b.data != nullptr);
        assert(out.rows > 0);
        assert(out.cols > 0);
        assert(a.cols == b.rows);
        assert(out.rows == a.rows && out.cols == b.cols);

        for (int i = 0; i < a.rows; ++i)
        {
            for (int j = 0; j < b.cols; ++j)
            {
                double accumulated = 0.0;
                for (int k = 0; k < a.cols; ++k)
                {
                    accumulated += a.data[i * a.cols + k] * b.data[k * b.cols + j];
                }
                out.data[i * out.cols + j] = accumulated;
            }
        }

        return;
    }
}
