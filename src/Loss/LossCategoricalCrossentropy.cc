#include "LossCategoricalCrossEntropy.h"

void NEURAL_NETWORK::LossCategoricalCrossEntropy::forward(const Eigen::MatrixXd& predictions, 
																	 const Eigen::MatrixXi& targets)
{
	int samples = predictions.rows();

	Eigen::MatrixXd y_pred_clipped = predictions.array().max(1e-7).min(1-1e-7);
	Eigen::VectorXd correct_confidences(samples);

	if (targets.cols() == 1) 
	{
		for (int i = 0; i < samples; i++) 
		{
			correct_confidences(i) = y_pred_clipped(i, targets(i, 0));
		}
	} 
	else if(targets.cols() == 2)
	{
		correct_confidences = (y_pred_clipped.array() * (targets.cast<double>()).array()).rowwise().sum();
	}

	output_ = -correct_confidences.array().log();
}

void NEURAL_NETWORK::LossCategoricalCrossEntropy::backward(const Eigen::MatrixXd& d_values, 
																	  const Eigen::MatrixXi& targets)
{
	int samples = d_values.rows();
	int labels = d_values.cols();

	Eigen::MatrixXd y_true;

	if (targets.cols() == 1) 
	{
		Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(labels, labels);
		
		y_true = Eigen::MatrixXd::Zero(samples, labels);
		for (int i = 0; i < samples; i++) 
		{
			int target_class = targets(i, 0);
			y_true.row(i) = identity.row(target_class);
		}
	}
	else if (targets.cols() == 2) 
	{
		y_true = targets.cast<double>();
	} 

	d_inputs_ = -y_true.array() / d_values.array();

	d_inputs_ /= samples;
}