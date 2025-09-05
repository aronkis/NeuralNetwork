#include "LayerDropout.h"

NEURAL_NETWORK::LayerDropout::LayerDropout(double rate)
{
	rate_ = 1 - rate;
}

void NEURAL_NETWORK::LayerDropout::forward(const Eigen::MatrixXd& inputs, 
										   bool training)
{
	inputs_ = inputs;

	if (!training)
	{
		output_ = inputs_;
		mask_.resize(inputs_.rows(), inputs_.cols());
		mask_.setOnes();
		return;
	}

	mask_ = (Eigen::MatrixXd::Random(inputs_.rows(), inputs_.cols()).array() + 1) / 2;
	mask_ = (mask_.array() < rate_).cast<double>() / rate_;
	output_ = inputs_.array() * mask_.array();
}

void NEURAL_NETWORK::LayerDropout::backward(const Eigen::MatrixXd& dvalues)
{
	d_inputs_ = dvalues.array() * mask_.array();
}

Eigen::MatrixXd NEURAL_NETWORK::LayerDropout::predictions() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDropout::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDropout::GetDInput() const
{
	return d_inputs_;
}

double NEURAL_NETWORK::LayerDropout::GetRate() const 
{ 
	return rate_; 
}

void NEURAL_NETWORK::LayerDropout::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}