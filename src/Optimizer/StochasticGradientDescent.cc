#include "StochasticGradientDescent.h"

NEURAL_NETWORK::StochasticGradientDescent::StochasticGradientDescent(double learning_rate, 
																	 double decay, 
																	 double momentum) : Optimizer(learning_rate, decay)
{
	momentum_ = momentum;
}

void NEURAL_NETWORK::StochasticGradientDescent::UpdateParameters(NEURAL_NETWORK::LayerBase& layer)
{
	Eigen::Tensor<double, 2> weight_update;
	Eigen::Tensor<double, 1> bias_update;

	weight_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
											  layer.GetWeights().dimension(1));
	bias_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	if (momentum_ > 0.0)
	{
		// Initialize momentums if they don't exist
		if (layer.GetWeightMomentums().dimension(0) == 0 || layer.GetWeightMomentums().dimension(1) == 0)
		{
			Eigen::Tensor<double, 2> weight_momentums(layer.GetWeights().dimension(0),
													  layer.GetWeights().dimension(1));
			weight_momentums.setZero();
			layer.SetWeightMomentums(weight_momentums);

			Eigen::Tensor<double, 1> bias_momentums(layer.GetBiases().dimension(0));
			bias_momentums.setZero();
			layer.SetBiasMomentums(bias_momentums);
		}

		// Vectorized momentum updates
		weight_update = layer.GetWeightMomentums() * momentum_ - layer.GetDWeights() * current_learning_rate_;
		bias_update = layer.GetBiasMomentums() * momentum_ - layer.GetDBiases() * current_learning_rate_;

		layer.SetWeightMomentums(weight_update);
		layer.SetBiasMomentums(bias_update);
	}
	else
	{
		// Simple gradient descent without momentum - vectorized
		weight_update = layer.GetDWeights() * (-current_learning_rate_);
		bias_update = layer.GetDBiases() * (-current_learning_rate_);
	}

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}

double NEURAL_NETWORK::StochasticGradientDescent::GetMomentum() const
{ 
	return momentum_; 
}
