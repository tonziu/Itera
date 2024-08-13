#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <genetics/evolution.h>
#include <network/neuralnetwork.h>
#include <game/pong.h>

void init_network(network::NeuralNetwork &network)
{
    // network.Add_Layer(network::DenseLayer(6, 1, math::matrix_sigmoid_in_place));
    network.Add_Layer(network::DenseLayer(6, 1, math::matrix_tanh_in_place));
}

int main(void)
{
    int num_population = 1000;
    int num_parents = 200;

    std::vector<network::NeuralNetwork> networks(num_population);
    for (auto& network : networks)
    {
        init_network(network);
    }

    while (true)
    {
        std::vector<double> scores = genetics::evaluate(networks);
        std::vector<network::NeuralNetwork> parents = genetics::selection(networks, scores, num_parents);
        std::vector<network::NeuralNetwork> children = genetics::crossingover(parents);
        genetics::mutation(children, 0.01, 5);
        genetics::evolve(networks, children, scores);

        std::cout << "Median score: " << math::median(scores) << "\n";
    }

    return 0;
}