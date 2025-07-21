#include "ActivationReLU.h"

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;
	output_ = inputs.cwiseMax(0.0);
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::MatrixXd& d_values)
{
	dinput_ = d_values;
	dinput_.array() *= (inputs_.array() > 0.0).cast<double>().array();
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetDInput() const
{
	return dinput_;
}