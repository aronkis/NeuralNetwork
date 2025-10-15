#include "RMSProp.h"
#include "Helpers.h"

NEURAL_NETWORK::RMSProp::RMSProp(double learning_rate, 
								 double decay, 
								 double epsilon, 
								 double rho): Optimizer(learning_rate, decay) 
{
	epsilon_ = epsilon;
	rho_ = rho;
}

double NEURAL_NETWORK::RMSProp::GetRho() const
{
	return rho_;
}

double NEURAL_NETWORK::RMSProp::GetEpsilon() const
{
	return epsilon_;
}

void NEURAL_NETWORK::RMSProp::UpdateParameters(NEURAL_NETWORK::LayerBase& layer)
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

	// Update caches manually
	weight_cache_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
													layer.GetWeights().dimension(1));
	bias_cache_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Vectorized cache updates
	weight_cache_update = layer.GetWeightCaches() * rho_ + layer.GetDWeights().square() * (1 - rho_);
	bias_cache_update = layer.GetBiasCaches() * rho_ + layer.GetDBiases().square() * (1 - rho_);

	layer.SetWeightCaches(weight_cache_update);
	layer.SetBiasCaches(bias_cache_update);

	// Calculate parameter updates
	weight_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
											  layer.GetWeights().dimension(1));
	bias_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Vectorized parameter updates
	weight_update = layer.GetDWeights() * (-current_learning_rate_) / (layer.GetWeightCaches() + epsilon_).sqrt();
	bias_update = layer.GetDBiases() * (-current_learning_rate_) / (layer.GetBiasCaches() + epsilon_).sqrt();

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}