#include "Loss.h"

void NEURAL_NETWORK::Loss::CalculateLoss(const Eigen::Tensor<double, 2>& predictions,
										 const Eigen::Tensor<double, 2>& targets,
										 bool include_regularization)
{
	forward(predictions, targets);

	// Calculate mean manually for tensor
	double sum = 0.0;
	int count = 0;
	for (int i = 0; i < output_.dimension(0); i++)
	{
		for (int j = 0; j < output_.dimension(1); j++)
		{
			sum += output_(i, j);
			count++;
		}
	}
	loss_ = sum / count;

	accumulated_loss_ += sum;
	accumulated_count_ += output_.dimension(0);

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

void NEURAL_NETWORK::Loss::RememberTrainableLayers(const std::vector<std::weak_ptr<LayerBase>>& layers)
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
				// Calculate absolute sum manually for tensors
				double abs_sum = 0.0;
				const auto& weights = layer->GetWeights();
				for (int i = 0; i < weights.dimension(0); i++)
				{
					for (int j = 0; j < weights.dimension(1); j++)
					{
						abs_sum += std::abs(weights(i, j));
					}
				}
				regularization_loss_ += layer->GetWeightRegularizerL1() * abs_sum;
			}

			if (layer->GetWeightRegularizerL2() > 0)
			{
				// Calculate squared sum manually for tensors
				double squared_sum = 0.0;
				const auto& weights = layer->GetWeights();
				for (int i = 0; i < weights.dimension(0); i++)
				{
					for (int j = 0; j < weights.dimension(1); j++)
					{
						squared_sum += weights(i, j) * weights(i, j);
					}
				}
				regularization_loss_ += layer->GetWeightRegularizerL2() * squared_sum;
			}

			if (layer->GetBiasRegularizerL1() > 0)
			{
				// Calculate absolute sum manually for bias tensors
				double abs_sum = 0.0;
				const auto& biases = layer->GetBiases();
				for (int i = 0; i < biases.dimension(0); i++)
				{
					abs_sum += std::abs(biases(i));
				}
				regularization_loss_ += layer->GetBiasRegularizerL1() * abs_sum;
			}

			if (layer->GetBiasRegularizerL2() > 0)
			{
				// Calculate squared sum manually for bias tensors
				double squared_sum = 0.0;
				const auto& biases = layer->GetBiases();
				for (int i = 0; i < biases.dimension(0); i++)
				{
					squared_sum += biases(i) * biases(i);
				}
				regularization_loss_ += layer->GetBiasRegularizerL2() * squared_sum;
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

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Loss::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Loss::GetDInput() const
{
	return d_inputs_;
}