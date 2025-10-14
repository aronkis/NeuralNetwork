#ifndef __ADA_GRAD_H__
#define __ADA_GRAD_H__

#include "Optimizer.h"

namespace NEURAL_NETWORK
{
	class AdaGrad : public Optimizer
	{
	public:
		AdaGrad(double learning_rate = 0.01, 
				double decay = 0.0, 
				double epsilon = 1e-7);
		~AdaGrad() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerBase& layer) override;

		double GetEpsilon() const;

	private:
		double epsilon_;
	};
} // namespace NEURAL_NETWORK

#endif // __ADA_GRAD_H__