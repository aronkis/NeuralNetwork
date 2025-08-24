#include "ActivationReLU.h"

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::MatrixXd& inputs, bool training)
{
	(void)training; // unused parameter
	inputs_ = inputs;
	output_ = inputs.cwiseMax(0.0);
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_ = d_values;
	d_inputs_.array() *= (inputs_.array() > 0.0).cast<double>().array();
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationReLU::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationReLU::predictions() const
{
	return output_;
}
