#include "ActivationReLU.h"

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::Tensor<double, 2>& inputs,
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
			output_(r, c) = std::max(0.0, inputs(r, c));
		}
	}
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::Tensor<double, 2>& d_values)
{
	int rows = d_values.dimension(0);
	int cols = d_values.dimension(1);

	d_inputs_ = Eigen::Tensor<double, 2>(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			d_inputs_(r, c) = d_values(r, c) * (inputs_(r, c) > 0.0 ? 1.0 : 0.0);
		}
	}
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationReLU::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationReLU::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationReLU::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::ActivationReLU::predictions() const
{
	return output_;
}
