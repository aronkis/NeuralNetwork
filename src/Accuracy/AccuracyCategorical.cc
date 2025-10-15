#include "AccuracyCategorical.h"

Eigen::Tensor<double, 1> NEURAL_NETWORK::AccuracyCategorical::compare(const Eigen::Tensor<double, 2>& predictions,
															const Eigen::Tensor<double, 2>& targets) const
{
	int rows = targets.dimension(0);
	int cols = targets.dimension(1);

	// Convert targets to integer representation
	std::vector<int> targets_int(rows);

	if (cols > 1)
	{
		// One-hot encoded targets: find argmax for each row
		for (int i = 0; i < rows; i++)
		{
			int max_idx = 0;
			double max_val = targets(i, 0);
			for (int j = 1; j < cols; j++)
			{
				if (targets(i, j) > max_val)
				{
					max_val = targets(i, j);
					max_idx = j;
				}
			}
			targets_int[i] = max_idx;
		}
	}
	else
	{
		// Single column targets: cast to int
		for (int i = 0; i < rows; i++)
		{
			targets_int[i] = static_cast<int>(targets(i, 0));
		}
	}

	// Determine number of classes
	int num_classes = 0;
	if (cols > 1)
	{
		num_classes = cols;
	}
	else
	{
		// Find max label value
		double max_label = targets(0, 0);
		for (int i = 0; i < rows; i++)
		{
			if (targets(i, 0) > max_label)
				max_label = targets(i, 0);
		}
		num_classes = static_cast<int>(max_label) + 1;
		if (num_classes < 2) num_classes = 2;
	}

	// Get predicted classes
	std::vector<int> pred_classes(rows);
	int pred_cols = predictions.dimension(1);

	if (pred_cols > 1)
	{
		// Multi-class predictions: find argmax for each row
		for (int i = 0; i < rows; i++)
		{
			int max_idx = 0;
			double max_val = predictions(i, 0);
			for (int j = 1; j < pred_cols; j++)
			{
				if (predictions(i, j) > max_val)
				{
					max_val = predictions(i, j);
					max_idx = j;
				}
			}
			pred_classes[i] = max_idx;
		}
	}
	else
	{
		// Single column predictions
		if (num_classes > 2)
		{
			for (int i = 0; i < rows; i++)
			{
				pred_classes[i] = static_cast<int>(predictions(i, 0));
			}
		}
		else
		{
			// Binary classification
			double min_val = predictions(0, 0);
			double max_val = predictions(0, 0);
			for (int i = 0; i < rows; i++)
			{
				if (predictions(i, 0) < min_val) min_val = predictions(i, 0);
				if (predictions(i, 0) > max_val) max_val = predictions(i, 0);
			}

			if (min_val >= 0.0 && max_val <= 1.0)
			{
				// Probability output
				for (int i = 0; i < rows; i++)
				{
					pred_classes[i] = predictions(i, 0) > 0.5 ? 1 : 0;
				}
			}
			else
			{
				// Raw output
				for (int i = 0; i < rows; i++)
				{
					pred_classes[i] = static_cast<int>(predictions(i, 0));
				}
			}
		}
	}

	// Compare predictions with targets
	Eigen::Tensor<double, 1> result(rows);
	for (int i = 0; i < rows; i++)
	{
		result(i) = (pred_classes[i] == targets_int[i]) ? 1.0 : 0.0;
	}

	return result;
}
