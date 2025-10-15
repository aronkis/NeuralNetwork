#include  "Accuracy.h"

void NEURAL_NETWORK::Accuracy::Calculate(const Eigen::Tensor<double, 2>& predictions,
										 const Eigen::Tensor<double, 2>& targets)
{
	Eigen::Tensor<double, 1> comparisons_tensor = compare(predictions, targets);

	// Convert tensor to array for mean calculation
	int size = comparisons_tensor.dimension(0);
	Eigen::ArrayXd comparisons(size);
	for (int i = 0; i < size; i++)
	{
		comparisons(i) = comparisons_tensor(i);
	}

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