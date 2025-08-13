#include "Loss.h"

void NEURAL_NETWORK::Loss::CalculateLoss(const Eigen::MatrixXd& predictions, 
										 const Eigen::MatrixXi& targets)
{
	forward(predictions, targets);
	loss_ = output_.array().mean();
}

double NEURAL_NETWORK::Loss::RegularizationLoss(LayerDense& layer) const
{
	double regularization_loss = 0.0;

	if (layer.GetWeightRegularizerL1() > 0)
	{
		regularization_loss += layer.GetWeightRegularizerL1() * layer.GetWeights().array().abs().sum();
	}

	if (layer.GetWeightRegularizerL2() > 0)
	{
		regularization_loss += layer.GetWeightRegularizerL2() * layer.GetWeights().array().square().sum();
	}

	if (layer.GetBiasRegularizerL1() > 0)
	{
		regularization_loss += layer.GetBiasRegularizerL1() * layer.GetBiases().array().abs().sum();
	}

	if (layer.GetBiasRegularizerL2() > 0)
	{
		regularization_loss += layer.GetBiasRegularizerL2() * layer.GetBiases().array().square().sum();
	}

	return regularization_loss;
}

double NEURAL_NETWORK::Loss::GetLoss() const
{
	return loss_;
}   


const Eigen::MatrixXd& NEURAL_NETWORK::Loss::GetOutput() const
{
	return output_;
}
const Eigen::MatrixXd& NEURAL_NETWORK::Loss::GetDInput() const
{
	return d_inputs_;
}