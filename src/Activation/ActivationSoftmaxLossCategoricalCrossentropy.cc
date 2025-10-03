#include "ActivationSoftmaxLossCategoricalCrossentropy.h"
#include "LossCategoricalCrossentropy.h"

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::forward(const Eigen::MatrixXd& inputs, 
																		   bool training)
{
	softmax_.forward(inputs, training);
	output_ = softmax_.GetOutput();
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::backward(const Eigen::MatrixXd& dvalues)
{
	if (targets_.rows() > 0) 
	{
		int samples = dvalues.rows();
		
		Eigen::VectorXi y_true(samples);
		if (targets_.cols() > 1) 
		{
			for (int i = 0; i < samples; i++) 
			{
				Eigen::Index idx;
				targets_.row(i).maxCoeff(&idx);
				y_true(i) = static_cast<int>(idx);
			}
		} 
		else
		{
			y_true = targets_.col(0);
		}
		
		d_inputs_ = dvalues;
		
		for (int i = 0; i < samples; i++) 
		{
			d_inputs_(i, y_true(i)) -= 1.0;
		}
		
		d_inputs_ /= samples;
	} 
	else 
	{
		d_inputs_ = dvalues;
	}
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::predictions() const
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

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::storeTargets(const Eigen::MatrixXi& targets) 
{
	targets_ = targets;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::GetOutput() const 
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::GetDInput() const 
{
	return d_inputs_;
}