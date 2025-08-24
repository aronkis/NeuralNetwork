#include  "Accuracy.h"

double NEURAL_NETWORK::Accuracy::Calculate(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& targets)
{
	Eigen::ArrayXd comparisons = compare(predictions, targets);
	return comparisons.mean();
}