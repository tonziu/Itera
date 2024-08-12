#ifndef H_ITERA_DENSE_LAYER_H
#define H_ITERA_DENSE_LAYER_H

#include <functional>
#include <math/matrix.h>

using Matrix = math::Matrix;

namespace network
{
    class DenseLayer
    {
    private:
        Matrix weights;
        Matrix biases;
        Matrix output;
        std::function<void(Matrix&)> activation;
    public:
        DenseLayer(int input_size, int output_size, std::function<void(Matrix&)> activation)
            :activation(activation)
        {
            math::matrix_alloc(weights, input_size, output_size);
            math::matrix_alloc(biases, 1, output_size);
            math::matrix_alloc(output, 1, output_size);

            math::matrix_random_in_place(weights, -1.0, 1.0);
            math::matrix_random_in_place(biases, -1.0, 1.0);
        }

        void forward(Matrix& x)
        {
            matrix_prod(output, x, weights);
            matrix_add_in_place(output, biases);
            activation(output);
        }

        Matrix Get_Output() const
        {
            return output;
        }
    };
}

#endif // H_ITERA_DENSE_LAYER_H