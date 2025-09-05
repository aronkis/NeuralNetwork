#include "AdaGrad.h"
#include "Helpers.h"

NEURAL_NETWORK::AdaGrad::AdaGrad(double learning_rate, 
								 double decay, 
								 double epsilon) : Optimizer(learning_rate, decay)
{
	epsilon_ = epsilon;
}

void NEURAL_NETWORK::AdaGrad::UpdateParameters(NEURAL_NETWORK::LayerDense& layer)
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
	weight_cache_update = layer.GetDWeights().array().square().matrix();
	bias_cache_update = layer.GetDBiases().array().square().matrix();

	layer.UpdateWeightsCache(weight_cache_update);
	layer.UpdateBiasesCache(bias_cache_update);

	weight_update = -current_learning_rate_ * 
					layer.GetDWeights().array() / 
					(layer.GetWeightCaches().array().sqrt() + 
					 epsilon_);
	bias_update = -current_learning_rate_ * layer.GetDBiases().array() / 
				  (layer.GetBiasCaches().array().sqrt() +
				   epsilon_);

	layer.UpdateWeights(weight_update);
	layer.UpdateBiases(bias_update);
}