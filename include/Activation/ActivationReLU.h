#ifndef __ACTIVATION_RELU_H__
#define __ACTIVATION_RELU_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class ActivationReLU : public LayerBase
	{
	public:
		ActivationReLU() = default;
		~ActivationReLU() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;

		// Tensor interface implementation
		bool SupportsTensorInterface() const override;
		void forward(const Eigen::Tensor<double, 4>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 4>& dvalues) override;
		const Eigen::Tensor<double, 4>& GetTensorOutput() const override;
		const Eigen::Tensor<double, 4>& GetTensorDInput() const override;
		void SetTensorDInput(const Eigen::Tensor<double, 4>& dinput) override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		
		void SetDInput(const Eigen::MatrixXd& dinput) override;

	private:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;

		// Tensor versions for tensor interface
		Eigen::Tensor<double, 4> tensor_inputs_;
		Eigen::Tensor<double, 4> tensor_output_;
		Eigen::Tensor<double, 4> tensor_d_inputs_;
	};
} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_RELU_H__