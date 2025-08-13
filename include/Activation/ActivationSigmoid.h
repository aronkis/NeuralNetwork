#ifndef __ACTIVATION_SIGMOID_H__
#define __ACTIVATION_SIGMOID_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
	class ActivationSigmoid
	{
	public:
		void forward(const Eigen::MatrixXd& inputs);
		void backward(const Eigen::MatrixXd& dvalues);

		const Eigen::MatrixXd& GetOutput() const;
		const Eigen::MatrixXd& GetDInput() const;

	private:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};
}

#endif // __ACTIVATION_SIGMOID_H__