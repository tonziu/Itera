#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <network/neuralnetwork.h>

using namespace math;

int main(void)
{
    network::NeuralNetwork network;
    network::DenseLayer layer_a(2, 3, matrix_relu_in_place);
    network::DenseLayer layer_b(3, 1, matrix_tanh_in_place);
    
    network.Add_Layer(layer_a);
    network.Add_Layer(layer_b);

    // Initialize input matrix
    math::Matrix input;
    math::matrix_alloc(input, 1, 2);
    input.data[0] = 1.0;
    input.data[1] = -2.0;

    // Perform forward pass
    Matrix output;
    matrix_alloc(output, input.rows, layer_b.Get_Output().cols);
    
    output = network.forward(input);

    // Print the output matrix
    std::cout << "Output matrix:" << std::endl;
    matrix_print(output);

    return 0;
}