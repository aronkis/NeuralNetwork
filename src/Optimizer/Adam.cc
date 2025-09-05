#include "Adam.h"
#include "Helpers.h"

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

void NEURAL_NETWORK::Adam::UpdateParameters(NEURAL_NETWORK::LayerDense& layer)
{
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
