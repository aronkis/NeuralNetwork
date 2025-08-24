#ifndef __ACTIVATION_SIGMOID_H__
#define __ACTIVATION_SIGMOID_H__

#include <Eigen/Dense>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class ActivationSigmoid : public LayerBase
	{
	public:
		ActivationSigmoid() = default;
		~ActivationSigmoid() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		void SetDInput(const Eigen::MatrixXd& dinput) override;
		Eigen::MatrixXd predictions() const override;

	private:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};

} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_SIGMOID_H__