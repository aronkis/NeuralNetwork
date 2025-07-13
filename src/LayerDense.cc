#include <random>
#include "LayerDense.h"

NEURAL_NETWORK::LayerDense::LayerDense(int n_inputs, int n_neurons)
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
	// for (int i = 0; i < n_inputs; i++) {
	//     for (int j = 0; j < n_neurons; ++j) {
	//         weights(i, j) = 0.01 * d(gen);
	//     }
	// }

	biases_ = Eigen::RowVectorXd::Zero(n_neurons);
}

void NEURAL_NETWORK::LayerDense::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;
	output_ = (inputs * weights_).rowwise() + biases_;
}

void NEURAL_NETWORK::LayerDense::backward(const Eigen::MatrixXd& d_values)
{
	d_weights_ = inputs_.transpose() * d_values;
	d_biases_ = d_values.colwise().sum();
	d_inputs_ = d_values * weights_.transpose();
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetDInput() const
{
	return d_inputs_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetDWeights() const
{
	return d_weights_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetDBiases() const
{
	return d_biases_;
}
