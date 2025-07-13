#include "ActivationSoftmaxLossCategoricalCrossentropy.h"

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy::forward(const Eigen::MatrixXd& inputs, 
																		   const Eigen::MatrixXi& targets) 
{
	softmax_.forward(inputs);
	output_ = softmax_.GetOutput();
	loss_.calculateLoss(softmax_.GetOutput(), targets);
	loss_value_ = loss_.GetLoss();
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy::backward(const Eigen::MatrixXd& d_values, 
																			const Eigen::MatrixXi& targets) 
{
	int samples = d_values.rows();
	
	Eigen::VectorXi y_true;
	
	if (targets.cols() == 2) 
	{
		y_true = Eigen::VectorXi(samples);
		for (int i = 0; i < samples; i++) 
		{
			targets.row(i).maxCoeff(&y_true(i));
		}
	} 
	else
	{
		y_true = targets.col(0);
	}
	
	d_inputs_ = d_values;
	
	for (int i = 0; i < samples; i++) 
	{
		d_inputs_(i, y_true(i)) -= 1.0;
	}
	
	d_inputs_ /= samples;
}

double NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy::GetLoss() const 
{
	return loss_value_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy::GetOutput() const 
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy::GetDInputs() const 
{
	return d_inputs_;
}

