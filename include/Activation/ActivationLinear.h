#ifndef __ACTIVATION_LINEAR_H__
#define __ACTIVATION_LINEAR_H__

#include <Eigen/Dense>
#include "Activation.h"

namespace NEURAL_NETWORK
{
	class ActivationLinear : public Activation
	{
	public:
		ActivationLinear() = default;
		~ActivationLinear() = default;

		ActivationLinear(const ActivationLinear&) = delete;
		ActivationLinear& operator=(const ActivationLinear&) = delete;

		void forward(const Eigen::MatrixXd& inputs) override;
		void backward(const Eigen::MatrixXd& dvalues) override;

	private:
		// Removed redundant variables and functions as they are now in the base class.
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_LINEAR_H__