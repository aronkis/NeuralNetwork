#include "ActivationLinear.h"

void NEURAL_NETWORK::ActivationLinear::forward(const Eigen::MatrixXd& inputs, bool training)
{
	(void)training; // unused parameter
	inputs_ = inputs;
	output_ = inputs;
}

void NEURAL_NETWORK::ActivationLinear::backward(const Eigen::MatrixXd& dvalues)
{
	d_inputs_ = dvalues;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationLinear::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationLinear::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationLinear::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationLinear::predictions() const
{
	return output_;
}