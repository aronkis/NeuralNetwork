#include "ActivationSigmoid.h"

void NEURAL_NETWORK::ActivationSigmoid::forward(const Eigen::MatrixXd& inputs, bool training)
{
	(void)training; // unused parameter
	inputs_ = inputs;
	output_ = 1.0 / (1.0 + (-inputs).array().exp());
}

void NEURAL_NETWORK::ActivationSigmoid::backward(const Eigen::MatrixXd& dvalues)
{
	d_inputs_ = dvalues.array() * output_.array() * (1 - output_.array());
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSigmoid::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSigmoid::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationSigmoid::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationSigmoid::predictions() const
{
	return (output_.array() > 0.5).cast<double>();
}