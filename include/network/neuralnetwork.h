#ifndef H_ITERA_NEURAL_NETWORK_H
#define H_ITERA_NEURAL_NETWORK_H

#include <network/denselayer.h>
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

namespace network
{
    class NeuralNetwork
    {
    private:
        std::vector<DenseLayer> layers;

    public:
        NeuralNetwork() = default;
        
        // From Json file obtained through the 'Save_To_Json' method.
        NeuralNetwork(const std::string &filename)
        {
            std::ifstream file(filename);
            json network_json;
            file >> network_json;

            for (const auto &layer_json : network_json["layers"])
            {
                layers.push_back(DenseLayer(layer_json));
            }
        }

        void Add_Layer(DenseLayer layer)
        {
            layers.push_back(layer);
        }

        void forward(const math::Matrix &input, math::Matrix &output)
        {
            assert(output.rows == 1);
            assert(output.cols == this->Get_Output_Size());

            math::Matrix current_input = input;

            for (auto &layer : layers)
            {
                layer.forward(current_input);
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

        DenseLayer &Get_Layer(int index)
        {
            assert(index >= 0 && index < layers.size());

            return layers[index];
        }

        void Mutation(double rate, double strength)
        {
            for (auto &layer : layers)
            {
                layer.Mutation(rate, strength);
            }

            return;
        }

        static NeuralNetwork Crossingover(NeuralNetwork &a, NeuralNetwork &b)
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

        void Save_To_Json(const std::string &filename)
        {
            json network_json;

            for (auto &layer : layers)
            {
                json layer_json = layer.Serialize();
                network_json["layers"].push_back(layer_json);
            }

            std::ofstream file(filename);
            file << network_json.dump(4); // Pretty-print with an indent of 4 spaces
        }
    };
}

#endif //  H_ITERA_NEURAL_NETWORK_H