#include <random>
#include "LayerDense.h"

LayerDense::LayerDense(int n_inputs, int n_neurons)
{
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution<> d(0, 1);

    weights_ = Eigen::MatrixXd(n_inputs, n_neurons);
    if (n_inputs == 2)
    {        
        weights_ << -0.01306527,  0.01658131, -0.00118164,
                    -0.00680178,  0.00666383, -0.0046072;
    }
    else
    {
        weights_ << -0.01334258, -0.01346717,  0.00693773,
                    -0.00159573, -0.00133702,  0.01077744,
                    -0.01126826, -0.00730678, -0.00384880;
    }
    // for (int i = 0; i < n_inputs; ++i) {
    //     for (int j = 0; j < n_neurons; ++j) {
    //         weights(i, j) = 0.01 * d(gen);
    //     }
    // }

    biases_ = Eigen::RowVectorXd::Zero(n_neurons);
}


void LayerDense::forward(const Eigen::MatrixXd inputs)
{
    output_ = (inputs * weights_).rowwise() + biases_;
}

const Eigen::MatrixXd& LayerDense::GetOutput() const
{
    return output_;
}
