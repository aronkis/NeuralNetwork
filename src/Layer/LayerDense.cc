#include <random>
#include <cmath>
#include "LayerDense.h"

NEURAL_NETWORK::LayerDense::LayerDense(int n_inputs, int n_neurons,
									   double weight_regularizer_l1,
									   double weight_regularizer_l2,
									   double bias_regularizer_l1,
									   double bias_regularizer_l2)
{
	std::mt19937 gen(0);
	// He normal initialization
	const double he_std = std::sqrt(2.0 / static_cast<double>(n_inputs));
	std::normal_distribution<> he_dist(0.0, he_std);

	// Initialize weights tensor with He normal distribution
	weights_ = Eigen::Tensor<double, 2>(n_inputs, n_neurons);
	for (int i = 0; i < n_inputs; i++)
	{
		for (int j = 0; j < n_neurons; j++)
		{
			weights_(i, j) = he_dist(gen);
		}
	}

	weight_regularizer_l1_ = weight_regularizer_l1;
	weight_regularizer_l2_ = weight_regularizer_l2;
	bias_regularizer_l1_ = bias_regularizer_l1;
	bias_regularizer_l2_ = bias_regularizer_l2;

	// Initialize biases tensor with zeros
	biases_ = Eigen::Tensor<double, 1>(n_neurons);
	biases_.setZero();
}

void NEURAL_NETWORK::LayerDense::forward(const Eigen::Tensor<double, 2>& inputs,
										 bool training)
{
	inputs_ = inputs;
	int batch_size = inputs.dimension(0);
	int output_size = weights_.dimension(1);

	output_ = Eigen::Tensor<double, 2>(batch_size, output_size);

	// Use tensor contraction for matrix multiplication: output = inputs * weights
	Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};
	output_ = inputs.contract(weights_, contraction_dims);

	// Add biases using broadcasting (vectorized)
	Eigen::array<int, 2> broadcast_dims = {batch_size, 1};
	Eigen::Tensor<double, 2> biases_broadcasted = biases_.reshape(Eigen::array<int, 2>{1, output_size}).broadcast(broadcast_dims);
	output_ += biases_broadcasted;
}

void NEURAL_NETWORK::LayerDense::backward(const Eigen::Tensor<double, 2>& d_values)
{
	int batch_size = inputs_.dimension(0);
	int input_size = inputs_.dimension(1);
	int output_size = d_values.dimension(1);

	// Initialize gradient tensors
	d_weights_ = Eigen::Tensor<double, 2>(input_size, output_size);
	d_biases_ = Eigen::Tensor<double, 1>(output_size);
	d_inputs_ = Eigen::Tensor<double, 2>(batch_size, input_size);

	// Compute d_weights = inputs.transpose() * d_values using tensor contraction
	Eigen::array<Eigen::IndexPair<int>, 1> weights_contraction = {Eigen::IndexPair<int>(0, 0)};
	d_weights_ = inputs_.contract(d_values, weights_contraction);

	// Compute d_biases = sum of d_values across batch dimension using reduction
	Eigen::array<int, 1> reduction_dims = {0};
	d_biases_ = d_values.sum(reduction_dims);

	// Add regularization gradients using vectorized operations
	if (weight_regularizer_l1_ > 0)
	{
		d_weights_ += weights_.sign() * weight_regularizer_l1_;
	}

	if (weight_regularizer_l2_ > 0)
	{
		d_weights_ += weights_ * (2 * weight_regularizer_l2_);
	}

	if (bias_regularizer_l1_ > 0)
	{
		d_biases_ += biases_.sign() * bias_regularizer_l1_;
	}

	if (bias_regularizer_l2_ > 0)
	{
		d_biases_ += biases_ * (2 * bias_regularizer_l2_);
	}

	// Compute d_inputs = d_values * weights.transpose() using tensor contraction
	Eigen::array<Eigen::IndexPair<int>, 1> inputs_contraction = {Eigen::IndexPair<int>(1, 1)};
	d_inputs_ = d_values.contract(weights_, inputs_contraction);
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetWeights() const
{
	return weights_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::LayerDense::GetBiases() const
{
	return biases_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetDInput() const
{
	return d_inputs_;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::LayerDense::predictions() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetDWeights() const
{
	return d_weights_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::LayerDense::GetDBiases() const
{
	return d_biases_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetWeightMomentums() const
{
	return weight_momentums_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::LayerDense::GetBiasMomentums() const
{
	return bias_momentums_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDense::GetWeightCaches() const
{
	return weight_caches_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::LayerDense::GetBiasCaches() const
{
	return bias_caches_;
}

double NEURAL_NETWORK::LayerDense::GetWeightRegularizerL1() const
{
	return weight_regularizer_l1_;
}

double NEURAL_NETWORK::LayerDense::GetWeightRegularizerL2() const
{
	return weight_regularizer_l2_;
}

double NEURAL_NETWORK::LayerDense::GetBiasRegularizerL1() const
{
	return bias_regularizer_l1_;
}

double NEURAL_NETWORK::LayerDense::GetBiasRegularizerL2() const
{
	return bias_regularizer_l2_;
}

std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> NEURAL_NETWORK::LayerDense::GetParameters() const
{
    return std::make_pair(weights_, biases_);
}

void NEURAL_NETWORK::LayerDense::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

void NEURAL_NETWORK::LayerDense::SetWeightMomentums(const Eigen::Tensor<double, 2>& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::LayerDense::SetBiasMomentums(const Eigen::Tensor<double, 1>& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::LayerDense::SetWeightCaches(const Eigen::Tensor<double, 2>& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::LayerDense::SetBiasCaches(const Eigen::Tensor<double, 1>& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::LayerDense::UpdateWeights(Eigen::Tensor<double, 2>& weight_update)
{
	weights_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateWeightsCache(Eigen::Tensor<double, 2>& weight_update)
{
    weight_caches_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiases(Eigen::Tensor<double, 1>& bias_update)
{
	biases_ += bias_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiasesCache(Eigen::Tensor<double, 1>& bias_update)
{
    bias_caches_ += bias_update;
}

void NEURAL_NETWORK::LayerDense::SetParameters(const Eigen::Tensor<double, 2>& weights,
											   const Eigen::Tensor<double, 1>& biases)
{
	weights_ = weights;
	biases_ = biases;
}