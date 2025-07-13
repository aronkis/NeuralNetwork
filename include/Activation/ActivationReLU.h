#ifndef __ACTIVATION_RELU_H__
#define __ACTIVATION_RELU_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{

	class ActivationReLU
	{
	public:
		ActivationReLU() = default;
		~ActivationReLU() = default;

		ActivationReLU(const ActivationReLU&) = delete;
		ActivationReLU& operator=(const ActivationReLU&) = delete;

		void forward(const Eigen::MatrixXd& inputs);
		void backward(const Eigen::MatrixXd& d_values); 

		const Eigen::MatrixXd& GetOutput() const;
		const Eigen::MatrixXd& GetDInput() const;

	private:
		Eigen::MatrixXd inputs_;

		Eigen::MatrixXd output_;

		Eigen::MatrixXd d_inputs_;
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_RELU_H__a