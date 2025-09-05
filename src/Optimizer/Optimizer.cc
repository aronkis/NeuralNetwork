#include "Optimizer.h"

double NEURAL_NETWORK::Optimizer::GetLearningRate() const
{
	return current_learning_rate_;
}

NEURAL_NETWORK::Optimizer::Optimizer(double learning_rate, double decay)
{
	learning_rate_ = learning_rate;
	current_learning_rate_ = learning_rate;
	decay_ = decay;
	iterations_ = 0;
}

void NEURAL_NETWORK::Optimizer::PreUpdateParameters()
{
	if (decay_ > 0.0)
	{
		current_learning_rate_ = learning_rate_ * 
								 (1.0 / (1.0 + decay_ * iterations_));
	}
}

void NEURAL_NETWORK::Optimizer::PostUpdateParameters()
{
	iterations_++;
}

double NEURAL_NETWORK::Optimizer::GetDecay() const 
{ 
	return decay_; 
}

int NEURAL_NETWORK::Optimizer::GetIterations() const 
{ 
	return iterations_; 
}