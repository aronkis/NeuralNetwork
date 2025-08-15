#include "ActivationLinear.h"

void NEURAL_NETWORK::ActivationLinear::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;
	output_ = inputs;
}

void NEURAL_NETWORK::ActivationLinear::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_ = d_values;
}
