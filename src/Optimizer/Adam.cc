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

	Eigen::MatrixXd weight_momentum_update;
	Eigen::MatrixXd weight_momentum_corrected;
	Eigen::MatrixXd weight_cache_update;
	Eigen::MatrixXd weight_cache_corrected;
	Eigen::MatrixXd weight_update;
	Eigen::RowVectorXd bias_momentum_update;
	Eigen::RowVectorXd bias_momentum_corrected;
	Eigen::RowVectorXd bias_cache_update;
	Eigen::RowVectorXd bias_cache_corrected;
	Eigen::RowVectorXd bias_update;

	if (layer.GetWeightCaches().size() == 0)
	{
		layer.SetWeightCaches(Eigen::MatrixXd::Zero(layer.GetWeights().rows(),
													layer.GetWeights().cols()));
		layer.SetWeightMomentums(Eigen::MatrixXd::Zero(layer.GetWeights().rows(),
													   layer.GetWeights().cols()));
		layer.SetBiasCaches(Eigen::RowVectorXd::Zero(layer.GetBiases().size()));
		layer.SetBiasMomentums(Eigen::RowVectorXd::Zero(layer.GetBiases().size()));
	}

	weight_momentum_update = beta1_ * layer.GetWeightMomentums() + 
							 (1 - beta1_) * layer.GetDWeights();
	bias_momentum_update = beta1_ * layer.GetBiasMomentums() + 
						   (1 - beta1_) * layer.GetDBiases();

	layer.SetWeightMomentums(weight_momentum_update);
	layer.SetBiasMomentums(bias_momentum_update);

	weight_momentum_corrected = weight_momentum_update / 
								(1 - std::pow(beta1_, iterations_ + 1));
	bias_momentum_corrected = bias_momentum_update / 
							  (1 - std::pow(beta1_, iterations_ + 1));

	weight_cache_update = beta2_ * layer.GetWeightCaches() + 
						  (1 - beta2_) * layer.GetDWeights().array().square().matrix();
	bias_cache_update = beta2_ * layer.GetBiasCaches() + 
						(1 - beta2_) * layer.GetDBiases().array().square().matrix();

	layer.SetWeightCaches(weight_cache_update);
	layer.SetBiasCaches(bias_cache_update);

	weight_cache_corrected = weight_cache_update / 
							 (1 - std::pow(beta2_, iterations_ + 1));
	bias_cache_corrected = bias_cache_update / 
						   (1 - std::pow(beta2_, iterations_ + 1));

	weight_update = -current_learning_rate_ * 
					weight_momentum_corrected.array() / 
					(weight_cache_corrected.array().sqrt() + 
					 epsilon_);
	bias_update = -current_learning_rate_ * 
				  bias_momentum_corrected.array() / 
				  (bias_cache_corrected.array().sqrt() + 
				   epsilon_);

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
	Eigen::MatrixXd gamma = bn_layer.GetWeights();  // gamma (scale parameters)
	Eigen::RowVectorXd beta = bn_layer.GetBiases(); // beta (shift parameters)
	Eigen::MatrixXd d_gamma = bn_layer.GetDWeights();
	Eigen::RowVectorXd d_beta = bn_layer.GetDBiases();

	// Initialize or resize momentum and cache if needed (BatchNorm can resize dynamically)
	if (bn_gamma_momentum_.size() == 0 ||
		bn_gamma_momentum_.rows() != gamma.rows() ||
		bn_gamma_momentum_.cols() != gamma.cols() ||
		bn_beta_momentum_.size() != beta.size())
	{
		bn_gamma_momentum_ = Eigen::MatrixXd::Zero(gamma.rows(), gamma.cols());
		bn_beta_momentum_ = Eigen::RowVectorXd::Zero(beta.size());
		bn_gamma_cache_ = Eigen::MatrixXd::Zero(gamma.rows(), gamma.cols());
		bn_beta_cache_ = Eigen::RowVectorXd::Zero(beta.size());
	}
	
	// Update momentum (first moment)
	bn_gamma_momentum_ = beta1_ * bn_gamma_momentum_ + (1 - beta1_) * d_gamma;
	bn_beta_momentum_ = beta1_ * bn_beta_momentum_ + (1 - beta1_) * d_beta;
	
	// Update cache (second moment)
	bn_gamma_cache_ = beta2_ * bn_gamma_cache_ + (1 - beta2_) * d_gamma.array().square().matrix();
	bn_beta_cache_ = beta2_ * bn_beta_cache_ + (1 - beta2_) * d_beta.array().square().matrix();
	
	// Bias correction
	Eigen::MatrixXd gamma_momentum_corrected = bn_gamma_momentum_ / (1 - std::pow(beta1_, iterations_ + 1));
	Eigen::RowVectorXd beta_momentum_corrected = bn_beta_momentum_ / (1 - std::pow(beta1_, iterations_ + 1));
	Eigen::MatrixXd gamma_cache_corrected = bn_gamma_cache_ / (1 - std::pow(beta2_, iterations_ + 1));
	Eigen::RowVectorXd beta_cache_corrected = bn_beta_cache_ / (1 - std::pow(beta2_, iterations_ + 1));
	
	// Compute updates
	Eigen::MatrixXd gamma_update = -current_learning_rate_ * 
								   gamma_momentum_corrected.array() / 
								   (gamma_cache_corrected.array().sqrt() + epsilon_);
	Eigen::RowVectorXd beta_update = -current_learning_rate_ * 
									 beta_momentum_corrected.array() / 
									 (beta_cache_corrected.array().sqrt() + epsilon_);
	
	// Apply updates
	bn_layer.UpdateWeights(gamma_update);
	bn_layer.UpdateBiases(beta_update);
}
