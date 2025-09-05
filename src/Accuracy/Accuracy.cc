#include  "Accuracy.h"

void NEURAL_NETWORK::Accuracy::Calculate(const Eigen::MatrixXd& predictions, 
										 const Eigen::MatrixXd& targets)
{
	Eigen::ArrayXd comparisons = compare(predictions, targets);
	accuracy_ = comparisons.mean();

	accumulated_accuracy_ += comparisons.sum();
	accumulated_count_ += comparisons.size();
}

void NEURAL_NETWORK::Accuracy::CalculateAccumulated()
{
	accumulated_accuracy_ /= accumulated_count_;
}

void NEURAL_NETWORK::Accuracy::NewPass()
{
	accumulated_accuracy_ = 0.0;
	accumulated_count_ = 0;
}

double NEURAL_NETWORK::Accuracy::GetAccuracy() const
{
	return accuracy_;
}

double NEURAL_NETWORK::Accuracy::GetAccumulatedAccuracy() const
{
	return accumulated_accuracy_;
}