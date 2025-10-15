#ifndef __ACTIVATION_SIGMOID_H__
#define __ACTIVATION_SIGMOID_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class ActivationSigmoid : public LayerBase
	{
	public:
		ActivationSigmoid() = default;
		~ActivationSigmoid() = default;

		void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 2>& dvalues) override;
		Eigen::Tensor<double, 2> predictions() const override;

		const Eigen::Tensor<double, 2>& GetOutput() const override;
		const Eigen::Tensor<double, 2>& GetDInput() const override;

		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;

	private:
		Eigen::Tensor<double, 2> inputs_;
		Eigen::Tensor<double, 2> output_;
		Eigen::Tensor<double, 2> d_inputs_;
	};
} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_SIGMOID_H__