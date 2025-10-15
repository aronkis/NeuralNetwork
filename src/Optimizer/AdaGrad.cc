#include "AdaGrad.h"
#include "Helpers.h"

NEURAL_NETWORK::AdaGrad::AdaGrad(double learning_rate, 
								 double decay, 
								 double epsilon) : Optimizer(learning_rate, decay)
{
	epsilon_ = epsilon;
}

double NEURAL_NETWORK::AdaGrad::GetEpsilon() const
{
	return epsilon_;
}

void NEURAL_NETWORK::AdaGrad::UpdateParameters(NEURAL_NETWORK::LayerBase& layer)
{
	Eigen::Tensor<double, 2> weight_update;
	Eigen::Tensor<double, 2> weight_cache_update;
	Eigen::Tensor<double, 1> bias_update;
	Eigen::Tensor<double, 1> bias_cache_update;

	// Initialize caches if they don't exist
	if (layer.GetWeightCaches().dimension(0) == 0 || layer.GetWeightCaches().dimension(1) == 0)
	{
		Eigen::Tensor<double, 2> weight_caches(layer.GetWeights().dimension(0),
											   layer.GetWeights().dimension(1));
		weight_caches.setZero();
		layer.SetWeightCaches(weight_caches);

		Eigen::Tensor<double, 1> bias_caches(layer.GetBiases().dimension(0));
		bias_caches.setZero();
		layer.SetBiasCaches(bias_caches);
	}

	// Calculate squared gradients
	weight_cache_update = Eigen::Tensor<double, 2>(layer.GetDWeights().dimension(0),
													layer.GetDWeights().dimension(1));
	bias_cache_update = Eigen::Tensor<double, 1>(layer.GetDBiases().dimension(0));

	// Vectorized squared gradients
	weight_cache_update = layer.GetDWeights().square();
	bias_cache_update = layer.GetDBiases().square();

	layer.UpdateWeightsCache(weight_cache_update);
	layer.UpdateBiasesCache(bias_cache_update);

	// Calculate weight updates
	weight_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
											  layer.GetWeights().dimension(1));
	bias_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Vectorized parameter updates
	weight_update = layer.GetDWeights() * (-current_learning_rate_) / (layer.GetWeightCaches() + epsilon_).sqrt();
	bias_update = layer.GetDBiases() * (-current_learning_rate_) / (layer.GetBiasCaches() + epsilon_).sqrt();

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}