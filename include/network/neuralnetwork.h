#ifndef H_ITERA_NEURAL_NETWORK_H
#define H_ITERA_NEURAL_NETWORK_H

#include <network/denselayer.h>

namespace network
{
    class NeuralNetwork
    {
    private:
        std::vector<DenseLayer> layers;
    
    public:
        NeuralNetwork() = default;

        void Add_Layer(DenseLayer layer)
        {
            layers.push_back(layer);
        }

        void forward(const math::Matrix& input, math::Matrix& output)
        {
            assert(output.rows == 1);
            assert(output.cols == this->Get_Output_Size());

            math::Matrix current_input = input;
            
            // Iterate through each layer and apply it to the current input
            for (auto& layer : layers)
            {
                // Forward pass through the current layer
                layer.forward(current_input);
                // Update the current input to be the output of the layer
                current_input = layer.Get_Output();
            }
            
            math::matrix_copy(current_input, output);
        }

        int Get_Output_Size() const
        {
            return layers.back().Get_Output().cols;
        }

        bool Is_Empty() const
        {
            return layers.size() == 0;
        }

        int Num_Layers() const
        {
            return layers.size();
        }

        DenseLayer& Get_Layer(int index)
        {
            assert(index >= 0 && index < layers.size());

            return layers[index];
        }

        void Mutation(double rate, double strength)
        {
            for (auto& layer : layers)
            {
                layer.Mutation(rate, strength);
            }

            return;
        }

        static NeuralNetwork Crossingover(NeuralNetwork& a, NeuralNetwork& b)
        {
            assert(a.Num_Layers() == b.Num_Layers());

            NeuralNetwork network;

            for (int i = 0; i < a.Num_Layers(); ++i)
            {
                DenseLayer child_layer = DenseLayer::Crossingover(a.Get_Layer(i), b.Get_Layer(i));
                network.Add_Layer(child_layer);
            }

            return network;
        }
    };
}

#endif //  H_ITERA_NEURAL_NETWORK_H