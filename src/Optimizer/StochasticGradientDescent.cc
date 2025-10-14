#include "StochasticGradientDescent.h"

NEURAL_NETWORK::StochasticGradientDescent::StochasticGradientDescent(double learning_rate, 
																	 double decay, 
																	 double momentum) : Optimizer(learning_rate, decay)
{
	momentum_ = momentum;
}

void NEURAL_NETWORK::StochasticGradientDescent::UpdateParameters(NEURAL_NETWORK::LayerBase& layer)
{
	Eigen::MatrixXd weight_update;
	Eigen::RowVectorXd bias_update;

	if (momentum_ > 0.0)
	{
		if (layer.GetWeightMomentums().size() == 0)
		{
			layer.SetWeightMomentums(Eigen::MatrixXd::Zero(layer.GetWeights().rows(), 
														   layer.GetWeights().cols()));
			layer.SetBiasMomentums(Eigen::RowVectorXd::Zero(layer.GetBiases().rows(), 
															layer.GetBiases().cols()));
		}

		weight_update = momentum_ * layer.GetWeightMomentums() - 
						current_learning_rate_ * layer.GetDWeights();
		bias_update = momentum_ * layer.GetBiasMomentums() - 
					  current_learning_rate_ * layer.GetDBiases();

		layer.SetWeightMomentums(weight_update);
		layer.SetBiasMomentums(bias_update);
	}
	else
	{
		weight_update = -current_learning_rate_ * layer.GetDWeights();
		bias_update = -current_learning_rate_ * layer.GetDBiases();
	}

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}

double NEURAL_NETWORK::StochasticGradientDescent::GetMomentum() const
{ 
	return momentum_; 
}
