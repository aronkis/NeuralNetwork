#include "Loss.h"

void NEURAL_NETWORK::Loss::CalculateLoss(const Eigen::MatrixXd& predictions, 
										 const Eigen::MatrixXd& targets,
										 bool include_regularization)
{
	forward(predictions, targets);
	loss_ = output_.array().mean();

	accumulated_loss_ += output_.sum();
	accumulated_count_ += output_.rows();

	if (include_regularization)
	{
		RegularizationLoss();
	}
	else
	{
		regularization_loss_ = 0.0;
	}
}

void NEURAL_NETWORK::Loss::CalculateAccumulatedLoss(bool include_regularization)
{
	accumulated_loss_ /= accumulated_count_;

	if (include_regularization)
	{
		RegularizationLoss();
	}
	else
	{
		regularization_loss_ = 0.0;
	}
}

void NEURAL_NETWORK::Loss::NewPass()
{
	accumulated_loss_ = 0.0;
	accumulated_count_ = 0;
}

void NEURAL_NETWORK::Loss::RememberTrainableLayers(const std::vector<std::weak_ptr<LayerDense>>& layers)
{
	trainable_layers_ = layers;
}

void NEURAL_NETWORK::Loss::RegularizationLoss()
{
	regularization_loss_ = 0.0;
	for (auto& weak_layer: trainable_layers_)
	{
		if (auto layer = weak_layer.lock())
		{
			if (layer->GetWeightRegularizerL1() > 0)
			{
				regularization_loss_ += layer->GetWeightRegularizerL1() * 
										layer->GetWeights().array().abs().sum();
			}

			if (layer->GetWeightRegularizerL2() > 0)
			{
				regularization_loss_ += layer->GetWeightRegularizerL2() * 
										layer->GetWeights().array().square().sum();
			}

			if (layer->GetBiasRegularizerL1() > 0)
			{
				regularization_loss_ += layer->GetBiasRegularizerL1() * 
										layer->GetBiases().array().abs().sum();
			}

			if (layer->GetBiasRegularizerL2() > 0)
			{
				regularization_loss_ += layer->GetBiasRegularizerL2() * 
										layer->GetBiases().array().square().sum();
			}
		}
	}
}

const double NEURAL_NETWORK::Loss::GetLoss() const
{
	return loss_;
}   

const double NEURAL_NETWORK::Loss::GetAccumulatedLoss() const
{
	return accumulated_loss_;
}

const double NEURAL_NETWORK::Loss::GetRegularizationLoss() const
{
	return regularization_loss_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Loss::GetOutput() const
{
	return output_;
}
const Eigen::MatrixXd& NEURAL_NETWORK::Loss::GetDInput() const
{
	return d_inputs_;
}