#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
	class Activation
	{
	public:
		virtual ~Activation() = default;

		virtual void forward(const Eigen::MatrixXd& inputs) = 0;
		virtual void backward(const Eigen::MatrixXd& dvalues) = 0;

		const Eigen::MatrixXd& GetOutput() const { return output_; }
		const Eigen::MatrixXd& GetDInput() const { return d_inputs_; }

	protected:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};
}

#endif // __ACTIVATION_H__
