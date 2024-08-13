#ifndef H_ITERA_DENSE_LAYER_H
#define H_ITERA_DENSE_LAYER_H

#include <functional>
#include <math/matrix.h>

namespace network
{
    class DenseLayer
    {
    private:
        math::Matrix weights;
        math::Matrix biases;
        math::Matrix output;
        std::function<void(math::Matrix &)> activation;

    public:
        DenseLayer(int input_size, int output_size, std::function<void(math::Matrix &)> activation)
            : activation(activation)
        {
            math::matrix_alloc(weights, input_size, output_size);
            math::matrix_alloc(biases, 1, output_size);
            math::matrix_alloc(output, 1, output_size);

            math::matrix_random_in_place(weights, -1.0, 1.0);
            math::matrix_random_in_place(biases, -1.0, 1.0);
        }

        void forward(math::Matrix &x)
        {
            matrix_prod(output, x, weights);
            matrix_add_in_place(output, biases);
            activation(output);
        }

        math::Matrix Get_Output() const
        {
            return output;
        }

        math::Matrix& Get_Weights()
        {
            return weights;
        }

        math::Matrix& Get_Biases()
        {
            return biases;
        }

        std::function<void(math::Matrix &)> Get_Activation() const
        {
            return activation;
        }

        void Mutation(double rate, double strength)
        {
            math::matrix_mutation(weights, rate, strength);
            math::matrix_mutation(biases,  rate, strength);
        }

        static DenseLayer Crossingover(DenseLayer &a, DenseLayer &b)
        {
            assert(a.Get_Weights().rows == b.Get_Weights().rows);
            assert(a.Get_Weights().cols == b.Get_Weights().cols);
            assert(a.Get_Biases().cols == b.Get_Biases().cols);

            DenseLayer child(a.Get_Weights().rows, a.Get_Weights().cols, a.Get_Activation());

            matrix_crossover(a.Get_Weights(), b.Get_Weights(), child.Get_Weights());
            matrix_crossover(a.Get_Biases(), b.Get_Biases(), child.Get_Biases());

            return child;
        }
    };
}

#endif // H_ITERA_DENSE_LAYER_H