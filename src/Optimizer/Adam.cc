#include "Adam.h"
#include "Helpers.h"
#include "BatchNormalization.h"

NEURAL_NETWORK::Adam::Adam(double learning_rate, 
						   double decay, 
						   double beta1, 
						   double beta2, 
						   double epsilon): Optimizer(learning_rate, decay) 
{
	beta1_ = beta1;
	beta2_ = beta2;
	epsilon_ = epsilon;
}

void NEURAL_NETWORK::Adam::UpdateParameters(NEURAL_NETWORK::LayerBase& layer)
{
	// Special case for BatchNormalization layer
	auto bn_layer = dynamic_cast<NEURAL_NETWORK::BatchNormalization*>(&layer);
	if (bn_layer)
	{
		UpdateBatchNormalizationParameters(*bn_layer);
		return;
	}

	Eigen::Tensor<double, 2> weight_momentum_update;
	Eigen::Tensor<double, 2> weight_momentum_corrected;
	Eigen::Tensor<double, 2> weight_cache_update;
	Eigen::Tensor<double, 2> weight_cache_corrected;
	Eigen::Tensor<double, 2> weight_update;
	Eigen::Tensor<double, 1> bias_momentum_update;
	Eigen::Tensor<double, 1> bias_momentum_corrected;
	Eigen::Tensor<double, 1> bias_cache_update;
	Eigen::Tensor<double, 1> bias_cache_corrected;
	Eigen::Tensor<double, 1> bias_update;

	// Initialize caches and momentums if they don't exist
	if (layer.GetWeightCaches().dimension(0) == 0 || layer.GetWeightCaches().dimension(1) == 0)
	{
		Eigen::Tensor<double, 2> weight_caches(layer.GetWeights().dimension(0),
											   layer.GetWeights().dimension(1));
		weight_caches.setZero();
		layer.SetWeightCaches(weight_caches);

		Eigen::Tensor<double, 2> weight_momentums(layer.GetWeights().dimension(0),
												  layer.GetWeights().dimension(1));
		weight_momentums.setZero();
		layer.SetWeightMomentums(weight_momentums);

		Eigen::Tensor<double, 1> bias_caches(layer.GetBiases().dimension(0));
		bias_caches.setZero();
		layer.SetBiasCaches(bias_caches);

		Eigen::Tensor<double, 1> bias_momentums(layer.GetBiases().dimension(0));
		bias_momentums.setZero();
		layer.SetBiasMomentums(bias_momentums);
	}

	// Initialize tensors
	weight_momentum_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
													  layer.GetWeights().dimension(1));
	bias_momentum_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Update momentum (first moment) - vectorized operations
	weight_momentum_update = layer.GetWeightMomentums() * beta1_ + layer.GetDWeights() * (1 - beta1_);
	bias_momentum_update = layer.GetBiasMomentums() * beta1_ + layer.GetDBiases() * (1 - beta1_);

	layer.SetWeightMomentums(weight_momentum_update);
	layer.SetBiasMomentums(bias_momentum_update);

	// Bias correction for momentum
	weight_momentum_corrected = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
														 layer.GetWeights().dimension(1));
	bias_momentum_corrected = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	double momentum_correction = 1 - std::pow(beta1_, iterations_ + 1);
	weight_momentum_corrected = weight_momentum_update / momentum_correction;
	bias_momentum_corrected = bias_momentum_update / momentum_correction;

	// Update cache (second moment)
	weight_cache_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
													layer.GetWeights().dimension(1));
	bias_cache_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Update cache (second moment) - vectorized operations
	weight_cache_update = layer.GetWeightCaches() * beta2_ + layer.GetDWeights().square() * (1 - beta2_);
	bias_cache_update = layer.GetBiasCaches() * beta2_ + layer.GetDBiases().square() * (1 - beta2_);

	layer.SetWeightCaches(weight_cache_update);
	layer.SetBiasCaches(bias_cache_update);

	// Bias correction for cache
	weight_cache_corrected = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
													  layer.GetWeights().dimension(1));
	bias_cache_corrected = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	double cache_correction = 1 - std::pow(beta2_, iterations_ + 1);
	weight_cache_corrected = weight_cache_update / cache_correction;
	bias_cache_corrected = bias_cache_update / cache_correction;

	// Calculate final updates
	weight_update = Eigen::Tensor<double, 2>(layer.GetWeights().dimension(0),
											  layer.GetWeights().dimension(1));
	bias_update = Eigen::Tensor<double, 1>(layer.GetBiases().dimension(0));

	// Calculate final updates - vectorized operations
	weight_update = weight_momentum_corrected * (-current_learning_rate_) / (weight_cache_corrected + epsilon_).sqrt();
	bias_update = bias_momentum_corrected * (-current_learning_rate_) / (bias_cache_corrected + epsilon_).sqrt();

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}

double NEURAL_NETWORK::Adam::GetBeta1() const 
{ 
	return beta1_; 
}

double NEURAL_NETWORK::Adam::GetBeta2() const 
{ 
	return beta2_; 
}

double NEURAL_NETWORK::Adam::GetEpsilon() const 
{ 
	return epsilon_; 
}

void NEURAL_NETWORK::Adam::UpdateBatchNormalizationParameters(NEURAL_NETWORK::BatchNormalization& bn_layer)
{
	// Get current parameters and gradients
	const Eigen::Tensor<double, 2>& gamma = bn_layer.GetWeights();  // gamma (scale parameters)
	const Eigen::Tensor<double, 1>& beta = bn_layer.GetBiases(); // beta (shift parameters)
	const Eigen::Tensor<double, 2>& d_gamma = bn_layer.GetDWeights();
	const Eigen::Tensor<double, 1>& d_beta = bn_layer.GetDBiases();

	// Initialize or resize momentum and cache if needed (BatchNorm can resize dynamically)
	if (bn_gamma_momentum_.dimension(0) == 0 ||
		bn_gamma_momentum_.dimension(0) != gamma.dimension(0) ||
		bn_gamma_momentum_.dimension(1) != gamma.dimension(1) ||
		bn_beta_momentum_.dimension(0) != beta.dimension(0))
	{
		bn_gamma_momentum_ = Eigen::Tensor<double, 2>(gamma.dimension(0), gamma.dimension(1));
		bn_gamma_momentum_.setZero();
		bn_beta_momentum_ = Eigen::Tensor<double, 1>(beta.dimension(0));
		bn_beta_momentum_.setZero();
		bn_gamma_cache_ = Eigen::Tensor<double, 2>(gamma.dimension(0), gamma.dimension(1));
		bn_gamma_cache_.setZero();
		bn_beta_cache_ = Eigen::Tensor<double, 1>(beta.dimension(0));
		bn_beta_cache_.setZero();
	}

	// Update momentum (first moment) - vectorized operations
	bn_gamma_momentum_ = bn_gamma_momentum_ * beta1_ + d_gamma * (1 - beta1_);
	bn_beta_momentum_ = bn_beta_momentum_ * beta1_ + d_beta * (1 - beta1_);

	// Update cache (second moment) - vectorized operations
	bn_gamma_cache_ = bn_gamma_cache_ * beta2_ + d_gamma.square() * (1 - beta2_);
	bn_beta_cache_ = bn_beta_cache_ * beta2_ + d_beta.square() * (1 - beta2_);

	// Bias correction
	Eigen::Tensor<double, 2> gamma_momentum_corrected(gamma.dimension(0), gamma.dimension(1));
	Eigen::Tensor<double, 1> beta_momentum_corrected(beta.dimension(0));
	Eigen::Tensor<double, 2> gamma_cache_corrected(gamma.dimension(0), gamma.dimension(1));
	Eigen::Tensor<double, 1> beta_cache_corrected(beta.dimension(0));

	double momentum_correction = 1 - std::pow(beta1_, iterations_ + 1);
	double cache_correction = 1 - std::pow(beta2_, iterations_ + 1);

	gamma_momentum_corrected = bn_gamma_momentum_ / momentum_correction;
	beta_momentum_corrected = bn_beta_momentum_ / momentum_correction;
	gamma_cache_corrected = bn_gamma_cache_ / cache_correction;
	beta_cache_corrected = bn_beta_cache_ / cache_correction;

	// Compute updates
	Eigen::Tensor<double, 2> gamma_update(gamma.dimension(0), gamma.dimension(1));
	Eigen::Tensor<double, 1> beta_update(beta.dimension(0));

	// Compute updates - vectorized operations
	gamma_update = gamma_momentum_corrected * (-current_learning_rate_) / (gamma_cache_corrected + epsilon_).sqrt();
	beta_update = beta_momentum_corrected * (-current_learning_rate_) / (beta_cache_corrected + epsilon_).sqrt();

	// Apply updates
	bn_layer.UpdateWeights(gamma_update);
	bn_layer.UpdateBiases(beta_update);
}
