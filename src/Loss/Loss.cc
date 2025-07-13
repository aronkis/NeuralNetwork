#include "Loss.h"

void NEURAL_NETWORK::Loss::calculateLoss(const Eigen::MatrixXd& predictions, 
										 const Eigen::MatrixXi& targets)
{
	Eigen::MatrixXd sample_loss = forward(predictions, targets);
	loss_ = sample_loss.array().mean();
}

double NEURAL_NETWORK::Loss::GetLoss() const
{
	return loss_;
}   