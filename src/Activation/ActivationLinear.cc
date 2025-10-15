#include "ActivationLinear.h"

void NEURAL_NETWORK::ActivationLinear::forward(const Eigen::Tensor<double, 2>& inputs,
											   bool training)
{
	inputs_ = inputs;
	output_ = inputs;
}

void NEURAL_NETWORK::ActivationLinear::backward(const Eigen::Tensor<double, 2>& dvalues)
{
	d_inputs_ = dvalues;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationLinear::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationLinear::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationLinear::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::ActivationLinear::predictions() const
{
	return output_;
}