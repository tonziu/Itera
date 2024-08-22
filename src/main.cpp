#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <genetics/evolution.h>
#include <network/neuralnetwork.h>
#include <game/pong.h>
#include <common.h>

void init_network(network::NeuralNetwork &network)
{
    network.Add_Layer(network::DenseLayer(6, 24, "tanh"));
    network.Add_Layer(network::DenseLayer(24, 6, "sigmoid"));
    network.Add_Layer(network::DenseLayer(6, 1, "tanh"));
}

void step(std::vector<network::NeuralNetwork>& networks, std::vector<double>& scores, int num_parents, common::SimInfo& info)
{
    info.num_generation++;
    genetics::evaluate(networks, scores, info);
    std::vector<network::NeuralNetwork> parents = genetics::selection(networks, scores, num_parents);
    std::vector<network::NeuralNetwork> children = genetics::crossingover(parents);
    genetics::mutation(children, 0.05, 10);
    genetics::evolve(networks, children, scores);
    networks[math::argmax(scores)].Save_To_Json("best_network.json");
}

int main(void)
{
    InitWindow(GAME_WIDTH, GAME_HEIGHT, "Itera: Pong v0.1");
    SetTargetFPS(0);

    int num_population = 1000;
    int num_parents = 500;

    common::SimInfo info = {}; 

    std::vector<network::NeuralNetwork> networks(num_population);
    for (auto &network : networks)
    {
        init_network(network);
    }

    std::vector<double> scores(num_population);

    while (!WindowShouldClose())
    {
        step(networks, scores, num_parents, info);
    }

    CloseWindow();
    return 0;
}