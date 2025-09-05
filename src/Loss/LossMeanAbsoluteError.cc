#include "LossMeanAbsoluteError.h"

void NEURAL_NETWORK::LossMeanAbsoluteError::forward(const Eigen::MatrixXd& predictions,
													const Eigen::MatrixXd& targets)
{
	output_ = (predictions.array() - targets.array()).abs().rowwise().mean();
}

void NEURAL_NETWORK::LossMeanAbsoluteError::backward(const Eigen::MatrixXd& d_values,
													 const Eigen::MatrixXd& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

	d_inputs_ = (targets.array() - d_values.array()).sign() / outputs;
	d_inputs_ /= samples;
}