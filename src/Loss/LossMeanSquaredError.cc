#include "LossMeanSquaredError.h"

void NEURAL_NETWORK::LossMeanSquaredError::forward(const Eigen::Tensor<double, 2>& predictions,
												   const Eigen::Tensor<double, 2>& targets)
{
	int rows = predictions.dimension(0);
	int cols = predictions.dimension(1);

	output_ = Eigen::Tensor<double, 2>(rows, 1);

	for (int r = 0; r < rows; r++)
	{
		double sum = 0.0;
		for (int c = 0; c < cols; c++)
		{
			double diff = predictions(r, c) - targets(r, c);
			sum += diff * diff;
		}
		output_(r, 0) = sum / cols;
	}
}

void NEURAL_NETWORK::LossMeanSquaredError::backward(const Eigen::Tensor<double, 2>& d_values,
													const Eigen::Tensor<double, 2>& targets)
{
	int samples = d_values.dimension(0);
	int outputs = d_values.dimension(1);

	d_inputs_ = Eigen::Tensor<double, 2>(samples, outputs);

	for (int r = 0; r < samples; r++)
	{
		for (int c = 0; c < outputs; c++)
		{
			d_inputs_(r, c) = 2.0 * (d_values(r, c) - targets(r, c)) / outputs / samples;
		}
	}
}
