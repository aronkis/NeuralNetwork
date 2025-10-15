#include "ActivationSoftmax.h"
#include <iostream>
#include <cmath>
#include <algorithm>

void NEURAL_NETWORK::ActivationSoftmax::forward(const Eigen::Tensor<double, 2>& inputs,
												bool training)
{
	inputs_ = inputs;
	int rows = inputs.dimension(0);
	int cols = inputs.dimension(1);

	output_ = Eigen::Tensor<double, 2>(rows, cols);

	// Apply softmax per row (sample)
	for (int r = 0; r < rows; r++)
	{
		// Find max for numerical stability
		double max_val = inputs(r, 0);
		for (int c = 1; c < cols; c++)
		{
			max_val = std::max(max_val, inputs(r, c));
		}

		// Calculate exp values and sum
		double sum_exp = 0.0;
		for (int c = 0; c < cols; c++)
		{
			output_(r, c) = std::exp(inputs(r, c) - max_val);
			sum_exp += output_(r, c);
		}

		// Normalize
		for (int c = 0; c < cols; c++)
		{
			output_(r, c) /= sum_exp;
		}
	}
}

void NEURAL_NETWORK::ActivationSoftmax::backward(const Eigen::Tensor<double, 2>& d_values)
{
	int rows = d_values.dimension(0);
	int cols = d_values.dimension(1);

	d_inputs_ = Eigen::Tensor<double, 2>(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		// Calculate dot product for this row
		double dot_product = 0.0;
		for (int c = 0; c < cols; c++)
		{
			dot_product += d_values(r, c) * output_(r, c);
		}

		// Calculate gradients
		for (int c = 0; c < cols; c++)
		{
			d_inputs_(r, c) = output_(r, c) * (d_values(r, c) - dot_product);
		}
	}
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSoftmax::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSoftmax::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationSoftmax::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::ActivationSoftmax::predictions() const
{
	int rows = output_.dimension(0);
	Eigen::Tensor<double, 2> preds(rows, 1);

	for (int r = 0; r < rows; r++)
	{
		// Find index of maximum value
		int max_idx = 0;
		double max_val = output_(r, 0);

		for (int c = 1; c < output_.dimension(1); c++)
		{
			if (output_(r, c) > max_val)
			{
				max_val = output_(r, c);
				max_idx = c;
			}
		}

		preds(r, 0) = static_cast<double>(max_idx);
	}

	return preds;
}