#include "ActivationSoftmax.h"
#include <iostream>

void NEURAL_NETWORK::ActivationSoftmax::forward(const Eigen::MatrixXd& inputs, 
												bool training)
{
	inputs_ = inputs;

	Eigen::MatrixXd stabilized = inputs.colwise() - inputs.rowwise().maxCoeff();
	Eigen::MatrixXd exp_values = stabilized.array().exp().matrix();

	output_.resizeLike(exp_values);
	output_ = exp_values.array().colwise() / exp_values.rowwise().sum().array();
}

void NEURAL_NETWORK::ActivationSoftmax::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_.resizeLike(d_values);

	const Eigen::MatrixXd dot = (d_values.array() * output_.array()).rowwise().sum();
	const Eigen::MatrixXd replicated = dot.replicate(1, output_.cols());
	d_inputs_ = (output_.array() * (d_values.array() - replicated.array())).matrix();
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
	for (Eigen::Index i = 0; i < output_.rows(); i++)
	{
		Eigen::Index idx;
		output_.row(i).maxCoeff(&idx);
		preds(i, 0) = static_cast<double>(idx);
	}
	return preds;
}