#ifndef __ACTIVATION_SOFTMAX_H__
#define __ACTIVATION_SOFTMAX_H__

#include <Eigen/Dense>
#include "Activation.h"

namespace NEURAL_NETWORK
{
	class ActivationSoftmax : public Activation
	{
	public:
		ActivationSoftmax() = default;
		~ActivationSoftmax() = default;

		ActivationSoftmax(const ActivationSoftmax&) = delete;
		ActivationSoftmax& operator=(const ActivationSoftmax&) = delete;

		void forward(const Eigen::MatrixXd& inputs) override;
		void backward(const Eigen::MatrixXd& dvalues) override;

	private:
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_SOFTMAX_H__