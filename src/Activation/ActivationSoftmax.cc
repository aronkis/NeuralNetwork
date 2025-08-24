#include "ActivationSoftmax.h"

void NEURAL_NETWORK::ActivationSoftmax::forward(const Eigen::MatrixXd& inputs, bool training)
{
	(void)training; // unused parameter
	inputs_ = inputs;

	// subtract row-wise max for numerical stability
	Eigen::MatrixXd stabilized = inputs.colwise() - inputs.rowwise().maxCoeff();
	Eigen::MatrixXd exp_values = stabilized.array().exp().matrix();
	// Normalize each row
	output_.resizeLike(exp_values);
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

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmax::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmax::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationSoftmax::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationSoftmax::predictions() const
{
	Eigen::MatrixXd preds(output_.rows(), 1);
	for (Eigen::Index i = 0; i < output_.rows(); ++i)
	{
		Eigen::Index idx;
		output_.row(i).maxCoeff(&idx);
		preds(i, 0) = static_cast<double>(idx);
	}
	return preds;
}