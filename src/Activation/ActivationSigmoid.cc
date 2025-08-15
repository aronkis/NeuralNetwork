#include "ActivationSigmoid.h"

void NEURAL_NETWORK::ActivationSigmoid::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;
	output_ = 1.0 / (1.0 + (-inputs).array().exp());
}

void NEURAL_NETWORK::ActivationSigmoid::backward(const Eigen::MatrixXd& dvalues)
{
	d_inputs_ = dvalues.array() * output_.array() * (1 - output_.array());
}
