#include "AccuracyRegression.h"

void NEURAL_NETWORK::AccuracyRegression::init(const Eigen::Tensor<double, 2>& target,
											  bool reinit)
{
	if (epsilon_ < 0.0 || reinit)
	{
		// Calculate mean of all elements in the tensor
		int rows = target.dimension(0);
		int cols = target.dimension(1);
		double sum = 0.0;
		int count = rows * cols;

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				sum += target(i, j);
			}
		}
		double mean = sum / count;

		// Calculate standard deviation
		double sum_squared_diff = 0.0;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				double diff = target(i, j) - mean;
				sum_squared_diff += diff * diff;
			}
		}
		double std_dev = std::sqrt(sum_squared_diff / count);
		epsilon_ = std_dev / 100.0;
	}
}

Eigen::Tensor<double, 1> NEURAL_NETWORK::AccuracyRegression::compare(const Eigen::Tensor<double, 2>& predictions,
														   const Eigen::Tensor<double, 2>& targets) const
{
	int rows = predictions.dimension(0);
	int cols = predictions.dimension(1);

	// Calculate per-row maximum absolute difference
	Eigen::Tensor<double, 1> per_row_max(rows);

	for (int r = 0; r < rows; r++)
	{
		double max_diff = 0.0;
		for (int c = 0; c < cols; c++)
		{
			double diff = std::abs(predictions(r, c) - targets(r, c));
			if (diff > max_diff)
			{
				max_diff = diff;
			}
		}
		per_row_max(r) = max_diff;
	}

	// Check if each row's max difference is within epsilon
	Eigen::Tensor<double, 1> result(rows);
	for (int r = 0; r < rows; r++)
	{
		result(r) = (per_row_max(r) < epsilon_) ? 1.0 : 0.0;
	}

	return result;
}

double NEURAL_NETWORK::AccuracyRegression::GetEpsilon() const 
{ 
	return epsilon_; 
}

