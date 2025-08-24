#ifndef __ADAM_H__
#define __ADAM_H__

#include "LayerDense.h"
#include "Optimizer.h"

namespace NEURAL_NETWORK
{
	class Adam : public Optimizer
	{
	public:
		Adam(double learning_rate = 0.001, double decay = 0.0, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
		~Adam() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerDense& layer) override;

	private:
		double beta1_;
		double beta2_;
		double epsilon_;
	};
} // namespace NEURAL_NETWORK

#endif // __ADAM_H__