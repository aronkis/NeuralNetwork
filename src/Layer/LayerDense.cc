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

	// Perform matrix multiplication: output = inputs * weights
	for (int batch = 0; batch < batch_size; batch++)
	{
		for (int out = 0; out < output_size; out++)
		{
			double sum = 0.0;
			for (int in = 0; in < inputs.dimension(1); in++)
			{
				sum += inputs(batch, in) * weights_(in, out);
			}
			output_(batch, out) = sum + biases_(out);
		}
	}
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

	// Compute d_weights = inputs.transpose() * d_values
	for (int in = 0; in < input_size; in++)
	{
		for (int out = 0; out < output_size; out++)
		{
			double sum = 0.0;
			for (int batch = 0; batch < batch_size; batch++)
			{
				sum += inputs_(batch, in) * d_values(batch, out);
			}
			d_weights_(in, out) = sum;
		}
	}

	// Compute d_biases = sum of d_values across batch dimension
	for (int out = 0; out < output_size; out++)
	{
		double sum = 0.0;
		for (int batch = 0; batch < batch_size; batch++)
		{
			sum += d_values(batch, out);
		}
		d_biases_(out) = sum;
	}

	// Add regularization gradients
	if (weight_regularizer_l1_ > 0)
	{
		for (int i = 0; i < input_size; i++)
		{
			for (int j = 0; j < output_size; j++)
			{
				d_weights_(i, j) += weight_regularizer_l1_ * (weights_(i, j) > 0 ? 1.0 : -1.0);
			}
		}
	}

	if (weight_regularizer_l2_ > 0)
	{
		for (int i = 0; i < input_size; i++)
		{
			for (int j = 0; j < output_size; j++)
			{
				d_weights_(i, j) += 2 * weight_regularizer_l2_ * weights_(i, j);
			}
		}
	}

	if (bias_regularizer_l1_ > 0)
	{
		for (int i = 0; i < output_size; i++)
		{
			d_biases_(i) += bias_regularizer_l1_ * (biases_(i) > 0 ? 1.0 : -1.0);
		}
	}

	if (bias_regularizer_l2_ > 0)
	{
		for (int i = 0; i < output_size; i++)
		{
			d_biases_(i) += 2 * bias_regularizer_l2_ * biases_(i);
		}
	}

	// Compute d_inputs = d_values * weights.transpose()
	for (int batch = 0; batch < batch_size; batch++)
	{
		for (int in = 0; in < input_size; in++)
		{
			double sum = 0.0;
			for (int out = 0; out < output_size; out++)
			{
				sum += d_values(batch, out) * weights_(in, out);
			}
			d_inputs_(batch, in) = sum;
		}
	}
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