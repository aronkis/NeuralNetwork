#include "AccuracyCategorical.h"

Eigen::ArrayXd NEURAL_NETWORK::AccuracyCategorical::compare(const Eigen::MatrixXd& predictions, 
															const Eigen::MatrixXd& targets) const
{
	Eigen::MatrixXi targets_int;

	if (targets.cols() > 1)
	{
		targets_int.resize(targets.rows(), 1);
		for (Eigen::Index i = 0; i < targets.rows(); i++)
		{
			Eigen::Index idx;
			targets.row(i).maxCoeff(&idx);
			targets_int(i, 0) = static_cast<int>(idx);
		}
	}
	else
	{
		targets_int = targets.cast<int>();
	}

	int num_classes = 0;

	if (targets.cols() > 1)
	{
		num_classes = static_cast<int>(targets.cols());
	}
	else
	{
		double max_label = targets.maxCoeff();
		num_classes = static_cast<int>(max_label) + 1;
		if (num_classes < 2) num_classes = 2;
	}

	Eigen::VectorXi pred_classes(predictions.rows());
	
	if (predictions.cols() > 1)
	{
		for (Eigen::Index i = 0; i < predictions.rows(); i++)
		{
			Eigen::Index idx;
			predictions.row(i).maxCoeff(&idx);
			pred_classes(i) = static_cast<int>(idx);
		}
	}
	else
	{
		if (num_classes > 2)
		{
			pred_classes = predictions.cast<int>().col(0);
		}
		else
		{
			double min_val = predictions.minCoeff();
			double max_val = predictions.maxCoeff();
			if (min_val >= 0.0 && max_val <= 1.0)
			{
				pred_classes = (predictions.array() > 0.5).cast<int>().matrix().col(0);
			}
			else
			{
				pred_classes = predictions.cast<int>().col(0);
			}
		}
	}

	return (pred_classes.array() == targets_int.col(0).array()).cast<double>();
}
