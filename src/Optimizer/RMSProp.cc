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

void NEURAL_NETWORK::RMSProp::UpdateParameters(NEURAL_NETWORK::LayerDense& layer)
{
	Eigen::MatrixXd weight_update;
	Eigen::MatrixXd weight_cache_update;
	Eigen::RowVectorXd bias_update;
	Eigen::RowVectorXd bias_cache_update;

	if (layer.GetWeightCaches().size() == 0) 
	{
		layer.SetWeightCaches(Eigen::MatrixXd::Zero(layer.GetWeights().rows(), 
													layer.GetWeights().cols()));
		layer.SetBiasCaches(Eigen::RowVectorXd::Zero(layer.GetBiases().size()));
	}

	weight_cache_update = rho_ * 
						  layer.GetWeightCaches() + 
						  (1 - rho_) * 
						  layer.GetDWeights().array().square().matrix();
	bias_cache_update = rho_ * 
						layer.GetBiasCaches() + 
						(1 - rho_) * 
						layer.GetDBiases().array().square().matrix();

	layer.SetWeightCaches(weight_cache_update);
	layer.SetBiasCaches(bias_cache_update);

	weight_update = -current_learning_rate_ * 
					layer.GetDWeights().array() / 
					(layer.GetWeightCaches().array().sqrt() + 
					 epsilon_);
	bias_update = -current_learning_rate_ * 
				  layer.GetDBiases().array() / 
				  (layer.GetBiasCaches().array().sqrt() + 
				   epsilon_);

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}