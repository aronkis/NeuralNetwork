#include "ActivationSoftmaxLossCategoricalCrossentropy.h"
#include "LossCategoricalCrossentropy.h"

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::forward(const Eigen::Tensor<double, 2>& inputs,
																		   bool training)
{
	softmax_.forward(inputs, training);
	output_ = softmax_.GetOutput();
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& dvalues)
{
	if (targets_.dimension(0) > 0)
	{
		int samples = dvalues.dimension(0);
		int cols = targets_.dimension(1);

		std::vector<int> y_true(samples);
		if (cols > 1)
		{
			for (int i = 0; i < samples; i++)
			{
				int max_idx = 0;
				int max_val = targets_(i, 0);
				for (int j = 1; j < cols; j++)
				{
					if (targets_(i, j) > max_val)
					{
						max_val = targets_(i, j);
						max_idx = j;
					}
				}
				y_true[i] = max_idx;
			}
		}
		else
		{
			for (int i = 0; i < samples; i++)
			{
				y_true[i] = targets_(i, 0);
			}
		}

		d_inputs_ = dvalues;

		for (int i = 0; i < samples; i++)
		{
			d_inputs_(i, y_true[i]) -= 1.0;
		}

		// Divide by samples
		int total_cols = d_inputs_.dimension(1);
		for (int i = 0; i < samples; i++)
		{
			for (int j = 0; j < total_cols; j++)
			{
				d_inputs_(i, j) /= samples;
			}
		}
	}
	else
	{
		d_inputs_ = dvalues;
	}
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::predictions() const
{
	int rows = output_.dimension(0);
	int cols = output_.dimension(1);
	Eigen::Tensor<double, 2> preds(rows, 1);

	for (int i = 0; i < rows; i++)
	{
		int max_idx = 0;
		double max_val = output_(i, 0);
		for (int j = 1; j < cols; j++)
		{
			if (output_(i, j) > max_val)
			{
				max_val = output_(i, j);
				max_idx = j;
			}
		}
		preds(i, 0) = static_cast<double>(max_idx);
	}
	return preds;
}

void NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::storeTargets(const Eigen::Tensor<int, 2>& targets)
{
	targets_ = targets;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy::GetDInput() const
{
	return d_inputs_;
}