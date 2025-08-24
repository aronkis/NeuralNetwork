#include "LayerDropout.h"

NEURAL_NETWORK::LayerDropout::LayerDropout(double rate)
{
	rate_ = 1 - rate;
}

void NEURAL_NETWORK::LayerDropout::forward(const Eigen::MatrixXd& inputs, bool training)
{
	inputs_ = inputs;

	// If not training, passthrough without dropout
	if (!training)
	{
		output_ = inputs_;
		// Keep mask consistent shape (all ones) in case it's accessed
		mask_.resize(inputs_.rows(), inputs_.cols());
		mask_.setOnes();
		return;
	}

	// Training: generate and apply scaled Bernoulli mask
	mask_ = (Eigen::MatrixXd::Random(inputs_.rows(), inputs_.cols()).array() + 1) / 2;
	mask_ = (mask_.array() < rate_).cast<double>() / rate_;
	output_ = inputs_.array() * mask_.array();
}

void NEURAL_NETWORK::LayerDropout::backward(const Eigen::MatrixXd& dvalues)
{
	d_inputs_ = dvalues.array() * mask_.array();
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDropout::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDropout::GetDInput() const
{
	return d_inputs_;
}

Eigen::MatrixXd NEURAL_NETWORK::LayerDropout::predictions() const
{
	return output_;
}
