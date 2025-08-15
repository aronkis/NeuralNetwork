#ifndef __ACTIVATION_SIGMOID_H__
#define __ACTIVATION_SIGMOID_H__

#include <Eigen/Dense>
#include "Activation.h"

namespace NEURAL_NETWORK
{
	class ActivationSigmoid : public Activation
	{
	public:
		ActivationSigmoid() = default;
		~ActivationSigmoid() = default;

		ActivationSigmoid(const ActivationSigmoid&) = delete;
		ActivationSigmoid& operator=(const ActivationSigmoid&) = delete;

		void forward(const Eigen::MatrixXd& inputs) override;
		void backward(const Eigen::MatrixXd& dvalues) override;

	private:
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_SIGMOID_H__