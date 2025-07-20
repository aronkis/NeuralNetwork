#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "LayerDense.h"

namespace NEURAL_NETWORK
{
	class Optimizer
	{
	public:
		Optimizer(double learning_rate, double decay);
		virtual ~Optimizer() = default;

		Optimizer(const Optimizer&) = delete;
		Optimizer& operator=(const Optimizer&) = delete;

		void PreUpdateParameters();
		virtual void UpdateParameters(NEURAL_NETWORK::LayerDense& layer) = 0;
		void PostUpdateParameters();

	protected:
		double learning_rate_;
		double current_learning_rate_;
		double decay_;
		int iterations_ = 0;
	};
} // namespace NEURAL_NETWORK

#endif // __OPTIMIZER_H__
