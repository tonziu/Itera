#ifndef H_ITERA_MATRIX_H
#define H_ITERA_MATRIX_H

#include <cassert>
#include <cstdlib>
#include <vector>
#include <random>
#include <ctime>
#include <nlohmann/json.hpp>

#include <math/functions.h>

using json = nlohmann::json;

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
    void matrix_add_in_place(Matrix &a, Matrix &b)
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
    void matrix_softmax(Matrix &out, Matrix &matrix)
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

    // Prints matrix data to the standard output.
    void matrix_print(const Matrix &matrix)
    {
        assert(matrix.data != nullptr);
        assert(matrix.rows > 0);
        assert(matrix.cols > 0);

        for (int i = 0; i < matrix.rows; ++i)
        {
            for (int j = 0; j < matrix.cols; ++j)
            {
                std::cout << matrix.data[i * matrix.cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void matrix_random_in_place(Matrix &m, double lower, double upper)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(lower, upper);

        for (int i = 0; i < m.rows * m.cols; ++i)
        {
            m.data[i] = dis(gen);
        }
    }

    // It applies a random point crossover.
    void matrix_crossover(const math::Matrix &parent1, const math::Matrix &parent2, math::Matrix &child)
    {
        assert(parent1.rows == parent2.rows && parent1.cols == parent2.cols);
        assert(child.rows == parent1.rows && child.cols == parent1.cols);

        for (int i = 0; i < parent1.rows; ++i)
        {
            for (int j = 0; j < parent1.cols; ++j)
            {
                // Alternate between parents based on index sum
                if ((i + j) % 2 == 0)
                {
                    child.data[i * child.cols + j] = parent1.data[i * parent1.cols + j];
                }
                else
                {
                    child.data[i * child.cols + j] = parent2.data[i * parent2.cols + j];
                }
            }
        }
    }

    // Apply random mutation according to a 'mutation_rate' and a 'mutation_strength'.
    void matrix_mutation(math::Matrix &m, double mutation_rate, double mutation_strength)
    {
        std::srand(static_cast<unsigned>(std::time(nullptr))); // Seed random number generator

        int total_elements = m.rows * m.cols;

        for (int i = 0; i < total_elements; ++i)
        {
            if (static_cast<double>(std::rand()) / RAND_MAX < mutation_rate)
            {
                double mutation = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2 * mutation_strength;
                m.data[i] += mutation;
            }
        }
    }

    json matrix_serialize(math::Matrix &m)
    {
        json matrix_json;
        matrix_json["rows"] = m.rows;
        matrix_json["cols"] = m.cols;

        // Convert the matrix data into a flat array
        matrix_json["data"] = std::vector<double>(m.data, m.data + m.rows * m.cols);

        return matrix_json;
    }

    math::Matrix matrix_deserialize(const json &matrix_json)
    {
        int rows = matrix_json["rows"];
        int cols = matrix_json["cols"];
        Matrix matrix;
        math::matrix_alloc(matrix, rows, cols);

        auto data = matrix_json["data"].get<std::vector<double>>();
        std::copy(data.begin(), data.end(), matrix.data);

        return matrix;
    }
}

#endif // H_ITERA_MATRIX_H
