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

        Matrix forward(const Matrix& input)
        {
            Matrix current_input = input;
            
            // Iterate through each layer and apply it to the current input
            for (auto& layer : layers)
            {
                // Forward pass through the current layer
                layer.forward(current_input);
                // Update the current input to be the output of the layer
                current_input = layer.Get_Output();
            }
            
            return current_input;
        }
    };
}

#endif //  H_ITERA_NEURAL_NETWORK_H