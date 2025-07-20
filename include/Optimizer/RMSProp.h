#ifndef __RMS_PROP_H__
#define __RMS_PROP_H__

#include "LayerDense.h"
#include "Optimizer.h"

namespace NEURAL_NETWORK
{
	class RMSProp : public Optimizer
	{
	public:
		RMSProp(double learning_rate = 0.01, double decay = 0.0, double epsilon = 1e-7, double rho = 0.9);
		~RMSProp() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerDense& layer) override;

	private:
		double epsilon_;
		double rho_;
	};
} // namespace NEURAL_NETWORK

#endif