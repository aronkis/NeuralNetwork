#include "ActivationSigmoid.h"
#include <cmath>

void NEURAL_NETWORK::ActivationSigmoid::forward(const Eigen::Tensor<double, 2>& inputs,
												bool training)
{
	inputs_ = inputs;
	int rows = inputs.dimension(0);
	int cols = inputs.dimension(1);

	output_ = Eigen::Tensor<double, 2>(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			output_(r, c) = 1.0 / (1.0 + std::exp(-inputs(r, c)));
		}
	}
}

void NEURAL_NETWORK::ActivationSigmoid::backward(const Eigen::Tensor<double, 2>& dvalues)
{
	int rows = dvalues.dimension(0);
	int cols = dvalues.dimension(1);

	d_inputs_ = Eigen::Tensor<double, 2>(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			d_inputs_(r, c) = dvalues(r, c) * output_(r, c) * (1.0 - output_(r, c));
		}
	}
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSigmoid::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSigmoid::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationSigmoid::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::ActivationSigmoid::predictions() const
{
	int rows = output_.dimension(0);
	int cols = output_.dimension(1);

	Eigen::Tensor<double, 2> predictions(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			predictions(r, c) = output_(r, c) > 0.5 ? 1.0 : 0.0;
		}
	}

	return predictions;
}