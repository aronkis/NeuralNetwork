#include <random>
#include "LayerDense.h"

// Uncomment the following lines to use constant weights for testing

NEURAL_NETWORK::LayerDense::LayerDense(int n_inputs, int n_neurons)
{
    std::mt19937 gen(0);
    std::normal_distribution<> d(-1, 1);
    weights_ = Eigen::MatrixXd(n_inputs, n_neurons);

    for (int i = 0; i < n_inputs; i++) 
    {
        for (int j = 0; j < n_neurons; j++) 
        {
            weights_(i, j) = 0.01 * d(gen);
        }
    }

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

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeights() const
{
	return weights_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiases() const
{
	return biases_;
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

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeightMomentums() const
{
	return weight_momentums_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiasMomentums() const
{
	return bias_momentums_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeightCaches() const
{
	return weight_caches_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiasCaches() const
{
	return bias_caches_;
}

void NEURAL_NETWORK::LayerDense::SetWeightMomentums(const Eigen::MatrixXd& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::LayerDense::SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::LayerDense::SetWeightCaches(const Eigen::MatrixXd& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::LayerDense::SetBiasCaches(const Eigen::RowVectorXd& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::LayerDense::UpdateWeights(Eigen::MatrixXd& weight_update)
{
	weights_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateWeightsCache(Eigen::MatrixXd& weight_update)
{
    weight_caches_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiases(Eigen::RowVectorXd& bias_update)
{
	biases_ += bias_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiasesCache(Eigen::RowVectorXd& bias_update)
{
    bias_caches_ += bias_update;
}