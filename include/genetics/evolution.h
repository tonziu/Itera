#ifndef H_ITERA_EVOLUTION_H
#define H_ITERA_EVOLUTION_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

#include <game/pong.h>
#include <network/neuralnetwork.h>
#include <common.h>

namespace genetics
{
    void evaluate(std::vector<network::NeuralNetwork> &networks, std::vector<double>& scores, common::SimInfo& info)
    {
        for (int i = 0; i < networks.size(); ++i)
        {
            game::Pong game(GAME_WIDTH, GAME_HEIGHT, networks[i]);
            scores[i] = game.Play(info);
        }
    }

    std::vector<network::NeuralNetwork> selection(std::vector<network::NeuralNetwork> &networks, std::vector<double> &scores, int num_selected)
    {
        assert(scores.size() == networks.size());
        assert(num_selected <= networks.size());

        std::vector<int> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&scores](int a, int b)
                  { return scores[a] > scores[b]; });

        // Elitism: directly add the best-performing networks
        std::vector<network::NeuralNetwork> selected_networks;
        int num_elites = num_selected / 10;
        for (int i = 0; i < num_elites; ++i)
        {
            selected_networks.push_back(networks[indices[i]]);
        }

        // Rank-based selection for the remaining slots
        std::vector<double> probabilities(scores.size());
        double total_rank = 0.0;

        for (int i = 0; i < scores.size(); ++i)
        {
            int rank = scores.size() - i;
            total_rank += rank;
            probabilities[i] = rank;
        }

        for (double &prob : probabilities)
        {
            prob /= total_rank;
        }

        std::default_random_engine generator;
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

        for (int i = num_elites; i < num_selected; ++i)
        {
            int selected_index = distribution(generator);
            selected_networks.push_back(networks[selected_index]);
        }

        return selected_networks;
    }

    std::vector<network::NeuralNetwork> crossingover(std::vector<network::NeuralNetwork> &parents)
    {
        assert(parents.size() % 2 == 0);

        std::vector<network::NeuralNetwork> children;
        for (int i = 0; i < parents.size() - 1; ++i)
        {
            network::NeuralNetwork child_network = network::NeuralNetwork::Crossingover(parents[i], parents[i + 1]);
            children.push_back(child_network);
        }

        return children;
    }

    void mutation(std::vector<network::NeuralNetwork> &networks, double rate, double strength)
    {
        for (auto &network : networks)
        {
            network.Mutation(rate, strength);
        }
    }

    void evolve(std::vector<network::NeuralNetwork> &networks, std::vector<network::NeuralNetwork> &children, std::vector<double> &scores)
    {
        assert(networks.size() == scores.size());
        assert(children.size() <= networks.size());

        std::vector<std::pair<double, int>> indexed_scores;
        for (int i = 0; i < scores.size(); ++i)
        {
            indexed_scores.emplace_back(scores[i], i);
        }

        std::sort(indexed_scores.begin(), indexed_scores.end());

        int num_replacements = children.size();
        for (int i = 0; i < num_replacements; ++i)
        {
            int index_to_replace = indexed_scores[i].second;
            networks[index_to_replace] = children[i];
        }

        return;
    }
}

#endif // H_ITERA_EVOLUTION_H