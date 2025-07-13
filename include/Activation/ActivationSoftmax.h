#ifndef __ACTIVATION_SOFTMAX_H__
#define __ACTIVATION_SOFTMAX_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
	class ActivationSoftmax
	{
	public:
		ActivationSoftmax() = default;
		~ActivationSoftmax() = default;

		ActivationSoftmax(const ActivationSoftmax&) = delete;
		ActivationSoftmax& operator=(const ActivationSoftmax&) = delete;

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

#endif // __ACTIVATION_SOFTMAX_H__