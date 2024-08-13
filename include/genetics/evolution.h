#ifndef H_ITERA_EVOLUTION_H
#define H_ITERA_EVOLUTION_H

#include <vector>
#include <game/pong.h>
#include <network/neuralnetwork.h>

namespace genetics
{
    std::vector<double> evaluate(std::vector<network::NeuralNetwork> networks)
    {
        std::vector<double> scores;

        for (auto& network : networks)
        {
            game::Pong game(400, 400, network);
            scores.push_back(game.Play(false));
        }

        return scores;
    }
}

#endif // H_ITERA_EVOLUTION_H