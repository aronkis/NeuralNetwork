#include "LayerDense.h"
#include <random>
#include <iostream>

LayerDense::LayerDense(int n_inputs, int n_neurons)
{
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution<> d(0, 1);

    weights = Eigen::MatrixXd(n_inputs, n_neurons);
    weights << -0.01306527,  0.01658131, -0.00118164,
               -0.00680178,  0.00666383, -0.0046072;
    // for (int i = 0; i < n_inputs; ++i) {
    //     for (int j = 0; j < n_neurons; ++j) {
    //         weights(i, j) = 0.01 * d(gen);
    //     }
    // }

    biases = Eigen::RowVectorXd::Zero(n_neurons);
}


void LayerDense::forward(Eigen::MatrixXd inputs)
{
    output = (inputs * weights).rowwise() + biases;
}

Eigen::MatrixXd LayerDense::GetOutput()
{
    return output;
}
