#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <genetics/evolution.h>
#include <network/neuralnetwork.h>
#include <game/pong.h>

void init_network(network::NeuralNetwork& network)
{
    network.Add_Layer(network::DenseLayer(6, 12, math::matrix_sigmoid_in_place));
    network.Add_Layer(network::DenseLayer(12, 1, math::matrix_tanh_in_place));
}

int main(void)
{
    int num_population = 10;
    std::vector<network::NeuralNetwork> networks(num_population);
    for (auto& network : networks)
    {
        init_network(network);
    }

    std::vector<double> scores = genetics::evaluate(networks);

    for (int i = 0; i < scores.size(); ++i)
    {
        std::cout << "Score #" << i+1 << ": " << scores[i] << "\n";
    }

    return 0;
}