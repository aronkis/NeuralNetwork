#include "ActivationSoftmax.h"

void NEURAL_NETWORK::ActivationSoftmax::forward(const Eigen::MatrixXd& inputs)
{
	inputs_ = inputs;

	Eigen::MatrixXd exp_values = (inputs.colwise() - (inputs.rowwise().maxCoeff())).array().exp();

	output_ = exp_values.array().colwise() / exp_values.rowwise().sum().array();
}

void NEURAL_NETWORK::ActivationSoftmax::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_ = Eigen::MatrixXd(d_values.rows(), d_values.cols());

	for (Eigen::Index i = 0; i < d_values.rows(); i++) 
	{
		Eigen::VectorXd single_output = output_.row(i).transpose();
		Eigen::MatrixXd jacobian_matrix = single_output.asDiagonal().toDenseMatrix() - single_output * single_output.transpose();

		d_inputs_.row(i) = (jacobian_matrix * d_values.row(i).transpose()).transpose();
	}
}