#include "LossMeanAbsoluteError.h"

void NEURAL_NETWORK::LossMeanAbsoluteError::forward(const Eigen::Tensor<double, 2>& predictions,
													const Eigen::Tensor<double, 2>& targets)
{
	int samples = predictions.dimension(0);
	int outputs = predictions.dimension(1);

	// Initialize output tensor
	if (output_.size() == 0 || output_.dimension(0) != samples || output_.dimension(1) != 1)
	{
		output_ = Eigen::Tensor<double, 2>(samples, 1);
	}

	// Calculate mean absolute error for each sample
	for (int s = 0; s < samples; s++)
	{
		double total_error = 0.0;
		for (int o = 0; o < outputs; o++)
		{
			total_error += std::abs(predictions(s, o) - targets(s, o));
		}
		output_(s, 0) = total_error / outputs;
	}
}

void NEURAL_NETWORK::LossMeanAbsoluteError::backward(const Eigen::Tensor<double, 2>& d_values,
													 const Eigen::Tensor<double, 2>& targets)
{
	int samples = d_values.dimension(0);
	int outputs = d_values.dimension(1);

	// Initialize d_inputs_ tensor
	if (d_inputs_.size() == 0 || d_inputs_.dimension(0) != samples || d_inputs_.dimension(1) != outputs)
	{
		d_inputs_ = Eigen::Tensor<double, 2>(samples, outputs);
	}

	// Calculate gradients manually for tensors
	for (int s = 0; s < samples; s++)
	{
		for (int o = 0; o < outputs; o++)
		{
			// Sign function for absolute error gradient
			double diff = targets(s, o) - d_values(s, o);
			double sign = (diff > 0) ? 1.0 : ((diff < 0) ? -1.0 : 0.0);
			d_inputs_(s, o) = sign / outputs / samples;
		}
	}
}