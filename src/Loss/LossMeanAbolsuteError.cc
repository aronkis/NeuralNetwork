#include "LossMeanAbsoluteError.h"

void NEURAL_NETWORK::LossMeanAbsoluteError::forward(const Eigen::MatrixXd& predictions,
													const Eigen::MatrixXi& targets)
{
	output_ = (predictions.array() - targets.cast<double>().array()).abs().rowwise().mean();
	loss_ = output_.array().mean();
}

void NEURAL_NETWORK::LossMeanAbsoluteError::forwardDouble(const Eigen::MatrixXd& predictions,
														   const Eigen::MatrixXd& targets)
{
	output_ = (predictions.array() - targets.array()).abs().rowwise().mean();
	loss_ = output_.array().mean();
}

void NEURAL_NETWORK::LossMeanAbsoluteError::backward(const Eigen::MatrixXd& d_values,
													 const Eigen::MatrixXi& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

	d_inputs_ = (targets.cast<double>().array() - d_values.array()).sign() / outputs;
	d_inputs_ /= samples;
}

void NEURAL_NETWORK::LossMeanAbsoluteError::backwardDouble(const Eigen::MatrixXd& d_values,
															const Eigen::MatrixXd& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

	d_inputs_ = (targets.array() - d_values.array()).sign() / outputs;
	d_inputs_ /= samples;
}
