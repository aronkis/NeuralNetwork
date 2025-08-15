#ifndef __ACTIVATION_RELU_H__
#define __ACTIVATION_RELU_H__

#include <Eigen/Dense>
#include "Activation.h"

namespace NEURAL_NETWORK
{
	class ActivationReLU : public Activation
	{
	public:
		ActivationReLU() = default;
		~ActivationReLU() = default;

		ActivationReLU(const ActivationReLU&) = delete;
		ActivationReLU& operator=(const ActivationReLU&) = delete;

		void forward(const Eigen::MatrixXd& inputs) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_RELU_H__