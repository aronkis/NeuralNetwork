#include "BatchNormalization.h"
#include <iostream>

NEURAL_NETWORK::BatchNormalization::BatchNormalization(int num_features,
													   double epsilon,
													   double momentum)
	: num_features_(num_features), epsilon_(epsilon), momentum_(momentum)
{
	gamma_ = Eigen::Tensor<double, 2>(1, num_features);
	gamma_.setConstant(1.0);

	beta_ = Eigen::Tensor<double, 1>(num_features);
	beta_.setZero();

	running_mean_ = Eigen::Tensor<double, 1>(num_features);
	running_mean_.setZero();

	running_var_ = Eigen::Tensor<double, 1>(num_features);
	running_var_.setConstant(1.0);

	d_gamma_ = Eigen::Tensor<double, 2>(1, num_features);
	d_gamma_.setZero();

	d_beta_ = Eigen::Tensor<double, 1>(num_features);
	d_beta_.setZero();
}

void NEURAL_NETWORK::BatchNormalization::forward(const Eigen::Tensor<double, 2>& inputs, bool training)
{
	batch_size_ = inputs.dimension(0);
	int features = inputs.dimension(1);
	cached_input_ = inputs;

	// Resize tensors if needed
	if (running_mean_.dimension(0) != features)
	{
		num_features_ = features;

		running_mean_ = Eigen::Tensor<double, 1>(features);
		running_mean_.setZero();

		running_var_ = Eigen::Tensor<double, 1>(features);
		running_var_.setConstant(1.0);

		gamma_ = Eigen::Tensor<double, 2>(1, features);
		gamma_.setConstant(1.0);

		beta_ = Eigen::Tensor<double, 1>(features);
		beta_.setZero();

		d_gamma_ = Eigen::Tensor<double, 2>(1, features);
		d_gamma_.setZero();

		d_beta_ = Eigen::Tensor<double, 1>(features);
		d_beta_.setZero();
	}

	// Initialize output and temporary tensors
	output_ = Eigen::Tensor<double, 2>(batch_size_, features);
	cached_normalized_ = Eigen::Tensor<double, 2>(batch_size_, features);
	cached_mean_ = Eigen::Tensor<double, 1>(features);
	cached_var_ = Eigen::Tensor<double, 1>(features);

	if (training)
	{
		// Calculate mean across batch dimension
		for (int f = 0; f < features; f++)
		{
			double sum = 0.0;
			for (int b = 0; b < batch_size_; b++)
			{
				sum += inputs(b, f);
			}
			cached_mean_(f) = sum / batch_size_;
		}

		// Calculate variance across batch dimension
		for (int f = 0; f < features; f++)
		{
			double sum_sq = 0.0;
			for (int b = 0; b < batch_size_; b++)
			{
				double diff = inputs(b, f) - cached_mean_(f);
				sum_sq += diff * diff;
			}
			cached_var_(f) = sum_sq / batch_size_;
		}

		// Normalize and store for backward pass
		for (int b = 0; b < batch_size_; b++)
		{
			for (int f = 0; f < features; f++)
			{
				double centered = inputs(b, f) - cached_mean_(f);
				double std_dev = std::sqrt(cached_var_(f) + epsilon_);
				cached_normalized_(b, f) = centered / std_dev;
			}
		}

		// Update running statistics
		for (int f = 0; f < features; f++)
		{
			running_mean_(f) = (1.0 - momentum_) * running_mean_(f) + momentum_ * cached_mean_(f);
			running_var_(f) = (1.0 - momentum_) * running_var_(f) + momentum_ * cached_var_(f);
		}
	}
	else
	{
		// Use running statistics for inference
		for (int b = 0; b < batch_size_; b++)
		{
			for (int f = 0; f < features; f++)
			{
				double centered = inputs(b, f) - running_mean_(f);
				double std_dev = std::sqrt(running_var_(f) + epsilon_);
				cached_normalized_(b, f) = centered / std_dev;
			}
		}
	}

	// Apply scale and shift: output = gamma * normalized + beta
	for (int b = 0; b < batch_size_; b++)
	{
		for (int f = 0; f < features; f++)
		{
			output_(b, f) = gamma_(0, f) * cached_normalized_(b, f) + beta_(f);
		}
	}
}

void NEURAL_NETWORK::BatchNormalization::backward(const Eigen::Tensor<double, 2>& dvalues)
{
	int features = dvalues.dimension(1);

	// Compute d_gamma: sum over batch dimension of (dvalues * normalized)
	for (int f = 0; f < features; f++)
	{
		double sum = 0.0;
		for (int b = 0; b < batch_size_; b++)
		{
			sum += dvalues(b, f) * cached_normalized_(b, f);
		}
		d_gamma_(0, f) = sum;
	}

	// Compute d_beta: sum over batch dimension of dvalues
	for (int f = 0; f < features; f++)
	{
		double sum = 0.0;
		for (int b = 0; b < batch_size_; b++)
		{
			sum += dvalues(b, f);
		}
		d_beta_(f) = sum;
	}

	// Compute gradients with respect to normalized values
	Eigen::Tensor<double, 2> d_normalized(batch_size_, features);
	for (int b = 0; b < batch_size_; b++)
	{
		for (int f = 0; f < features; f++)
		{
			d_normalized(b, f) = dvalues(b, f) * gamma_(0, f);
		}
	}

	// Initialize d_input tensor
	d_input_ = Eigen::Tensor<double, 2>(batch_size_, features);

	// Compute standard deviation and its inverse
	std::vector<double> std_dev(features);
	std::vector<double> inv_std_dev(features);
	for (int f = 0; f < features; f++)
	{
		std_dev[f] = std::sqrt(cached_var_(f) + epsilon_);
		inv_std_dev[f] = 1.0 / std_dev[f];
	}

	// Compute centered values (x - mean)
	Eigen::Tensor<double, 2> centered(batch_size_, features);
	for (int b = 0; b < batch_size_; b++)
	{
		for (int f = 0; f < features; f++)
		{
			centered(b, f) = cached_input_(b, f) - cached_mean_(f);
		}
	}

	// Compute d_variance
	std::vector<double> d_variance(features);
	for (int f = 0; f < features; f++)
	{
		double sum = 0.0;
		for (int b = 0; b < batch_size_; b++)
		{
			sum += d_normalized(b, f) * centered(b, f);
		}
		d_variance[f] = sum * (-0.5) * inv_std_dev[f] * inv_std_dev[f] * inv_std_dev[f];
	}

	// Compute d_mean (direct term)
	std::vector<double> d_mean_direct(features);
	for (int f = 0; f < features; f++)
	{
		double sum = 0.0;
		for (int b = 0; b < batch_size_; b++)
		{
			sum += d_normalized(b, f) * inv_std_dev[f];
		}
		d_mean_direct[f] = -sum;
	}

	// Compute d_mean (via variance term)
	std::vector<double> d_mean_via_var(features);
	for (int f = 0; f < features; f++)
	{
		double centered_sum = 0.0;
		for (int b = 0; b < batch_size_; b++)
		{
			centered_sum += centered(b, f);
		}
		d_mean_via_var[f] = d_variance[f] * (2.0 / batch_size_) * centered_sum;
	}

	// Total d_mean
	std::vector<double> d_mean(features);
	for (int f = 0; f < features; f++)
	{
		d_mean[f] = d_mean_direct[f] - d_mean_via_var[f];
	}

	// Compute final gradients
	for (int b = 0; b < batch_size_; b++)
	{
		for (int f = 0; f < features; f++)
		{
			double term1 = d_normalized(b, f) * inv_std_dev[f];
			double term2 = d_variance[f] * (2.0 / batch_size_) * centered(b, f);
			double term3 = d_mean[f] / batch_size_;
			d_input_(b, f) = term1 + term2 + term3;
		}
	}
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::BatchNormalization::GetWeights() const
{
	return gamma_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::BatchNormalization::GetBiases() const
{
	return beta_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::BatchNormalization::GetDWeights() const
{
	return d_gamma_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::BatchNormalization::GetDBiases() const
{
	return d_beta_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::BatchNormalization::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::BatchNormalization::GetDInput() const
{
	return d_input_;
}

void NEURAL_NETWORK::BatchNormalization::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_input_ = dinput;
}

void NEURAL_NETWORK::BatchNormalization::UpdateWeights(Eigen::Tensor<double, 2>& weight_update)
{
	gamma_ = gamma_ + weight_update;
}

void NEURAL_NETWORK::BatchNormalization::UpdateBiases(Eigen::Tensor<double, 1>& bias_update)
{
	beta_ = beta_ + bias_update;
}

int NEURAL_NETWORK::BatchNormalization::GetNumFeatures() const
{
	return num_features_;
}

std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> NEURAL_NETWORK::BatchNormalization::GetParameters() const
{
	// Pack gamma, running_mean, and running_var into the weights tensor
	// Row 0: gamma
	// Row 1: running_mean
	// Row 2: running_var
	Eigen::Tensor<double, 2> weights(3, num_features_);

	for (int f = 0; f < num_features_; f++)
	{
		weights(0, f) = gamma_(0, f);
		weights(1, f) = running_mean_(f);
		weights(2, f) = running_var_(f);
	}

	return std::make_pair(weights, beta_);
}

void NEURAL_NETWORK::BatchNormalization::SetParameters(const Eigen::Tensor<double, 2>& weights, const Eigen::Tensor<double, 1>& biases)
{
	// Check if this is the new format (3 rows) or old format (1 row)
	if (weights.dimension(0) == 3 && weights.dimension(1) == num_features_)
	{
		// New format: unpack gamma, running_mean, and running_var
		for (int f = 0; f < num_features_; f++)
		{
			gamma_(0, f) = weights(0, f);
			running_mean_(f) = weights(1, f);
			running_var_(f) = weights(2, f);
		}
	}
	else if (weights.dimension(0) == 1 && weights.dimension(1) == num_features_)
	{
		// Old format: only gamma (for backward compatibility)
		for (int f = 0; f < num_features_; f++)
		{
			gamma_(0, f) = weights(0, f);
		}
		// Keep existing running statistics or reset them
		std::cerr << "BatchNormalization::SetParameters warning: loading old format without running statistics" << std::endl;
	}
	else if (weights.dimension(0) == gamma_.dimension(0) && weights.dimension(1) == gamma_.dimension(1))
	{
		// Fallback for any other size match
		gamma_ = weights;
	}

	if (biases.dimension(0) == beta_.dimension(0))
	{
		beta_ = biases;
	}
}