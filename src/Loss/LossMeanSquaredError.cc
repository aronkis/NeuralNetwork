#include "LossMeanSquaredError.h"

void NEURAL_NETWORK::LossMeanSquaredError::forward(const Eigen::MatrixXd& predictions,
												   const Eigen::MatrixXi& targets)
{
	output_ = (predictions.array() - targets.cast<double>().array()).square().rowwise().mean();
	loss_ = output_.array().mean();
}

void NEURAL_NETWORK::LossMeanSquaredError::forwardDouble(const Eigen::MatrixXd& predictions,
												   const Eigen::MatrixXd& targets)
{
	output_ = (predictions.array() - targets.array()).square().rowwise().mean();
	loss_ = output_.array().mean();
}

void NEURAL_NETWORK::LossMeanSquaredError::backward(const Eigen::MatrixXd& d_values,
													const Eigen::MatrixXi& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

	d_inputs_ = 2.0 * (d_values.array() - targets.cast<double>().array()) / outputs;
	d_inputs_ /= samples;
}

void NEURAL_NETWORK::LossMeanSquaredError::backwardDouble(const Eigen::MatrixXd& d_values,
													const Eigen::MatrixXd& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

	d_inputs_ = 2.0 * (d_values.array() - targets.array()) / outputs;
	d_inputs_ /= samples;
}
