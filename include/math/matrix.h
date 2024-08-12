#ifndef H_ITERA_MATRIX_H
#define H_ITERA_MATRIX_H

#include <cassert>
#include <cstdlib>
#include <vector>

#include <math/functions.h>

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

    // Matrix element-wise addition (not in place).
    // The 'out' matrix should be already allocated using the right sizes.
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

    // Matrix product (not in place).
    // The 'out' matrix should be already allocated using the right sizes.
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

    // Applies RELU to a matrix (in place).
    void matrix_relu_in_place(Matrix &m)
    {
        assert(m.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            m.data[i] = relu(m.data[i]);
        }

        return;
    }

    // Applies RELU to a matrix (not in place).
    void matrix_relu(Matrix &out, Matrix &m)
    {
        assert(m.data != nullptr);
        assert(out.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);
        assert(out.rows == m.rows);
        assert(out.cols == m.cols);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            out.data[i] = relu(m.data[i]);
        }

        return;
    }

    // Applies sigmoid to a matrix (in place).
    void matrix_sigmoid_in_place(Matrix &m)
    {
        assert(m.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            m.data[i] = sigmoid(m.data[i]);
        }

        return;
    }

    // Applies sigmoid to a matrix (not in place).
    void matrix_sigmoid(Matrix &out, Matrix &m)
    {
        assert(m.data != nullptr);
        assert(out.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);
        assert(out.rows == m.rows);
        assert(out.cols == m.cols);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            out.data[i] = sigmoid(m.data[i]);
        }

        return;
    }

    // Applies tanh to a matrix (in place).
    void matrix_tanh_in_place(Matrix &m)
    {
        assert(m.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            m.data[i] = tanh(m.data[i]);
        }

        return;
    }

    // Applies tanh to a matrix (not in place).
    void matrix_tanh(Matrix &out, Matrix &m)
    {
        assert(m.data != nullptr);
        assert(out.data != nullptr);
        assert(m.rows > 0);
        assert(m.cols > 0);
        assert(out.rows == m.rows);
        assert(out.cols == m.cols);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            out.data[i] = tanh(m.data[i]);
        }

        return;
    }

    // Applies softmax to a matrix (in place).
    void matrix_softmax_in_place(Matrix &matrix)
    {
        assert(matrix.data != nullptr);
        assert(matrix.rows > 0);
        assert(matrix.cols > 0);

        std::vector<double> exp_values(matrix.rows * matrix.cols);
        double max_value = matrix.data[0];

        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            if (matrix.data[i] > max_value)
            {
                max_value = matrix.data[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            exp_values[i] = std::exp(matrix.data[i] - max_value);
            sum += exp_values[i];
        }

        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            matrix.data[i] = exp_values[i] / sum;
        }
    }

    // Applies softmax to a matrix (not in place).
    void matrix_softmax(Matrix& out, Matrix &matrix)
    {
        assert(matrix.data != nullptr);
        assert(out.data != nullptr);
        assert(matrix.rows > 0);
        assert(matrix.cols > 0);
        assert(out.rows == matrix.rows);
        assert(out.cols == matrix.cols);

        std::vector<double> exp_values(matrix.rows * matrix.cols);
        double max_value = matrix.data[0];

        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            if (matrix.data[i] > max_value)
            {
                max_value = matrix.data[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            exp_values[i] = std::exp(matrix.data[i] - max_value);
            sum += exp_values[i];
        }

        for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        {
            out.data[i] = exp_values[i] / sum;
        }
    }

    void matrix_print(const Matrix &matrix)
    {
        for (int i = 0; i < matrix.rows; ++i)
        {
            for (int j = 0; j < matrix.cols; ++j)
            {
                std::cout << matrix.data[i * matrix.cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

}

#endif // H_ITERA_MATRIX_H
