#include "AccuracyRegression.h"

void NEURAL_NETWORK::AccuracyRegression::init(const Eigen::MatrixXd& target, bool reinit)
{
	if (epsilon_ < 0.0 || reinit)
	{
    	double std_dev = std::sqrt((target.array() - target.mean()).square().mean());
		epsilon_ = std_dev / 250.0;
	}
}

Eigen::ArrayXd NEURAL_NETWORK::AccuracyRegression::compare(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& targets) const
{
	Eigen::VectorXd per_row_max = (predictions - targets).cwiseAbs().rowwise().maxCoeff();
	return (per_row_max.array() < epsilon_).cast<double>();
}
