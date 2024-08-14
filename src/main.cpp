#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <genetics/evolution.h>
#include <network/neuralnetwork.h>
#include <game/pong.h>

void init_network(network::NeuralNetwork &network)
{
    network.Add_Layer(network::DenseLayer(6, 8, "tanh"));
    network.Add_Layer(network::DenseLayer(8, 1, "tanh"));
}

void main_loop(bool train)
{
    if (!train)
    {
        network::NeuralNetwork best_network("best_network.json");
        game::Pong game(GAME_WIDTH, GAME_HEIGHT, best_network);
        game.Play(true);
        return;
    }

    int num_population = 1000;
    int num_parents = 400;

    std::vector<network::NeuralNetwork> networks(num_population);
    for (auto& network : networks)
    {
        init_network(network);
    }

    double best = 0;

    while (true)
    {
        std::vector<double> scores = genetics::evaluate(networks);
        std::vector<network::NeuralNetwork> parents = genetics::selection(networks, scores, num_parents);
        std::vector<network::NeuralNetwork> children = genetics::crossingover(parents);
        genetics::mutation(children, 0.01, 5);
        genetics::evolve(networks, children, scores);

        double median = math::median(scores);
        double curr_best = math::max(scores);

        std::cout << "\nMedian score: " << median << "\n";
        std::cout << "Maximum score: " << curr_best << "\n";
        std::cout << "Overall best: " << best << "\n\n";

        int best_score_index = math::argmax(scores);

        if (curr_best > best)
        {
            best = curr_best;
            game::Pong demo(GAME_WIDTH, GAME_HEIGHT, networks[best_score_index]);
            demo.Play(true);
            networks[best_score_index].Save_To_Json("best_network.json");
        }
    }

    return;
}

int main(void)
{
    bool train = 1;
    main_loop(train);
    return 0;
}