#include "LayerDropout.h"
#include <random>

NEURAL_NETWORK::LayerDropout::LayerDropout(double rate)
{
	rate_ = 1 - rate;
}

void NEURAL_NETWORK::LayerDropout::forward(const Eigen::Tensor<double, 2>& inputs,
										   bool training)
{
	inputs_ = inputs;
	int rows = inputs.dimension(0);
	int cols = inputs.dimension(1);

	output_ = Eigen::Tensor<double, 2>(rows, cols);
	mask_ = Eigen::Tensor<double, 2>(rows, cols);

	if (!training)
	{
		output_ = inputs_;
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				mask_(r, c) = 1.0;
			}
		}
		return;
	}

	// Generate random mask and apply dropout
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			double random_val = dis(gen);
			if (random_val < rate_)
			{
				mask_(r, c) = 1.0 / rate_;  // Scale up to maintain expected value
				output_(r, c) = inputs(r, c) * mask_(r, c);
			}
			else
			{
				mask_(r, c) = 0.0;
				output_(r, c) = 0.0;
			}
		}
	}
}

void NEURAL_NETWORK::LayerDropout::backward(const Eigen::Tensor<double, 2>& dvalues)
{
	int rows = dvalues.dimension(0);
	int cols = dvalues.dimension(1);
	d_inputs_ = Eigen::Tensor<double, 2>(rows, cols);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			d_inputs_(r, c) = dvalues(r, c) * mask_(r, c);
		}
	}
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::LayerDropout::predictions() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDropout::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerDropout::GetDInput() const
{
	return d_inputs_;
}

double NEURAL_NETWORK::LayerDropout::GetRate() const 
{ 
	return rate_; 
}

void NEURAL_NETWORK::LayerDropout::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_inputs_ = dinput;
}