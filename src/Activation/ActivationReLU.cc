#include "ActivationReLU.h"

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;
	output_ = inputs.cwiseMax(0.0);
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_ = d_values;
	d_inputs_.array() *= (inputs_.array() > 0.0).cast<double>().array();
}