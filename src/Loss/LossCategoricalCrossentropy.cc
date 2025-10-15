#include "LossCategoricalCrossentropy.h"

void NEURAL_NETWORK::LossCategoricalCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions,
														  const Eigen::Tensor<double, 2>& targets)
{
	int samples = predictions.dimension(0);
	int num_classes = predictions.dimension(1);

	// Initialize output tensor
	if (output_.size() == 0 || output_.dimension(0) != samples || output_.dimension(1) != 1)
	{
		output_ = Eigen::Tensor<double, 2>(samples, 1);
	}

	// Process each sample
	for (int i = 0; i < samples; i++)
	{
		double correct_confidence = 0.0;

		if (targets.dimension(1) == 1)  // Sparse labels
		{
			// Get the target class index
			int target_class = static_cast<int>(targets(i, 0));

			// Clip prediction to avoid log(0)
			double pred_clipped = std::max(1e-7, std::min(1.0 - 1e-7, predictions(i, target_class)));
			correct_confidence = pred_clipped;
		}
		else if (targets.dimension(1) > 1)  // One-hot encoded labels
		{
			// Sum of element-wise multiplication (dot product)
			for (int j = 0; j < num_classes; j++)
			{
				double pred_clipped = std::max(1e-7, std::min(1.0 - 1e-7, predictions(i, j)));
				correct_confidence += pred_clipped * targets(i, j);
			}
		}

		// Categorical cross entropy loss
		output_(i, 0) = -std::log(correct_confidence);
	}
}

void NEURAL_NETWORK::LossCategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& d_values,
														   const Eigen::Tensor<double, 2>& targets)
{
	int samples = d_values.dimension(0);
	int labels = d_values.dimension(1);

	// Initialize d_inputs_ tensor
	if (d_inputs_.size() == 0 || d_inputs_.dimension(0) != samples || d_inputs_.dimension(1) != labels)
	{
		d_inputs_ = Eigen::Tensor<double, 2>(samples, labels);
	}

	// Process gradients for each sample
	for (int i = 0; i < samples; i++)
	{
		if (targets.dimension(1) == 1)  // Sparse labels
		{
			// Convert sparse label to one-hot
			int target_class = static_cast<int>(targets(i, 0));

			for (int j = 0; j < labels; j++)
			{
				if (j == target_class)
				{
					// Gradient for correct class
					double clipped_pred = std::max(1e-7, std::min(1.0 - 1e-7, d_values(i, j)));
					d_inputs_(i, j) = -1.0 / clipped_pred;
				}
				else
				{
					// Gradient for incorrect classes
					d_inputs_(i, j) = 0.0;
				}
			}
		}
		else if (targets.dimension(1) > 1)  // One-hot encoded labels
		{
			// Direct computation for one-hot encoded targets
			for (int j = 0; j < labels; j++)
			{
				double clipped_pred = std::max(1e-7, std::min(1.0 - 1e-7, d_values(i, j)));
				d_inputs_(i, j) = -targets(i, j) / clipped_pred;
			}
		}
	}

	// Normalize by number of samples
	for (int i = 0; i < samples; i++)
	{
		for (int j = 0; j < labels; j++)
		{
			d_inputs_(i, j) /= samples;
		}
	}
}