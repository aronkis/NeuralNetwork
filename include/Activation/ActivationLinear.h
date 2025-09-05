#ifndef __ACTIVATION_LINEAR_H__
#define __ACTIVATION_LINEAR_H__

#include <Eigen/Dense>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class ActivationLinear : public LayerBase
	{
	public:
		ActivationLinear() = default;
		~ActivationLinear() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		
		void SetDInput(const Eigen::MatrixXd& dinput) override;
		
	private:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};
} // namespace NEURAL_NETWORK

#endif //__ACTIVATION_LINEAR_H__