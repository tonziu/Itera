#ifndef H_ITERA_DENSE_LAYER_H
#define H_ITERA_DENSE_LAYER_H

#include <functional>
#include <math/matrix.h>
#include <nlohmann/json.hpp>
#include <map>

namespace network
{
    class DenseLayer
    {
    private:
        math::Matrix weights;
        math::Matrix biases;
        math::Matrix output;
        std::function<void(math::Matrix &)> activation;
        std::string activation_name;

    public:
        DenseLayer(int input_size, int output_size, std::string activation_name)
            : activation_name(activation_name)
        {
            math::matrix_alloc(weights, input_size, output_size);
            math::matrix_alloc(biases, 1, output_size);
            math::matrix_alloc(output, 1, output_size);

            math::matrix_random_in_place(weights, -1.0, 1.0);
            math::matrix_random_in_place(biases, -1.0, 1.0);

            if (activation_name.compare("sigmoid") == 0)
            {
                activation = math::matrix_sigmoid_in_place;
            }
            else if (activation_name.compare("relu") == 0)
            {
                activation = math::matrix_relu_in_place;
            }
            else if (activation_name.compare("tanh") == 0)
            {
                activation = math::matrix_tanh_in_place;
            }
            else if (activation_name.compare("softmax") == 0)
            {
                activation = math::matrix_softmax_in_place;
            }
            else
            {
                std::cout << "Invalid activation function name.\n";
                exit(1);
            }
        }

        DenseLayer(const json& layer_json)
        {
            math::Matrix weights = math::matrix_deserialize(layer_json["weights"]);
            math::Matrix biases = math::matrix_deserialize(layer_json["biases"]);
            math::matrix_alloc(this->weights, weights.rows, weights.cols);
            math::matrix_alloc(this->biases, 1, biases.cols);
            math::matrix_alloc(output, 1, biases.cols);

            math::matrix_copy(weights, this->weights);
            math::matrix_copy(biases, this->biases);

            activation_name = layer_json["activation"];

            if (activation_name.compare("sigmoid") == 0)
            {
                activation = math::matrix_sigmoid_in_place;
            }
            else if (activation_name.compare("relu") == 0)
            {
                activation = math::matrix_relu_in_place;
            }
            else if (activation_name.compare("tanh") == 0)
            {
                activation = math::matrix_tanh_in_place;
            }
            else if (activation_name.compare("softmax") == 0)
            {
                activation = math::matrix_softmax_in_place;
            }
            else
            {
                std::cout << "Invalid activation function name.\n";
                exit(1);
            }
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

        math::Matrix &Get_Weights()
        {
            return weights;
        }

        math::Matrix &Get_Biases()
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
            math::matrix_mutation(biases, rate, strength);
        }

        json Serialize()
        {
            json layer_json;
            layer_json["weights"] = math::matrix_serialize(weights);
            layer_json["biases"] = math::matrix_serialize(biases);
            layer_json["activation"] = activation_name;

            return layer_json;
        }

        std::string Get_Activation_Name() const
        {
            return activation_name;
        }

        static DenseLayer Crossingover(DenseLayer &a, DenseLayer &b)
        {
            assert(a.Get_Weights().rows == b.Get_Weights().rows);
            assert(a.Get_Weights().cols == b.Get_Weights().cols);
            assert(a.Get_Biases().cols == b.Get_Biases().cols);

            DenseLayer child(a.Get_Weights().rows, a.Get_Weights().cols, a.Get_Activation_Name());

            matrix_crossover(a.Get_Weights(), b.Get_Weights(), child.Get_Weights());
            matrix_crossover(a.Get_Biases(), b.Get_Biases(), child.Get_Biases());

            return child;
        }
    };
}

#endif // H_ITERA_DENSE_LAYER_H