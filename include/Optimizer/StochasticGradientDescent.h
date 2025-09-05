#ifndef __STOCHASTIC_GRADIENT_DESCENT_H__
#define __STOCHASTIC_GRADIENT_DESCENT_H__

#include "Optimizer.h"

namespace NEURAL_NETWORK
{
	class StochasticGradientDescent : public Optimizer
	{
	public:
		StochasticGradientDescent(double learning_rate = 0.01, 
								  double decay = 0.0, 
								  double momentum = 0.0);
		~StochasticGradientDescent() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerDense& layer) override;

		double GetMomentum() const;

	private:
		double momentum_;
	};
} // namespace NEURAL_NETWORK

#endif // __STOCHASTIC_GRADIENT_DESCENT_H__