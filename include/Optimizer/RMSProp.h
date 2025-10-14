#ifndef __RMS_PROP_H__
#define __RMS_PROP_H__

#include "Optimizer.h"

namespace NEURAL_NETWORK
{
	class RMSProp : public Optimizer
	{
	public:
		RMSProp(double learning_rate = 0.01, 
				double decay = 0.0, 
				double epsilon = 1e-7, 
				double rho = 0.9);
		~RMSProp() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerBase& layer) override;

		double GetRho() const;
		double GetEpsilon() const;

	private:
		double rho_;
		double epsilon_;
	};
} // namespace NEURAL_NETWORK

#endif // __RMS_PROP_H__