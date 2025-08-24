#ifndef __LAYER_DROPOUT_H__
#define __LAYER_DROPOUT_H__

#include <Eigen/Dense>
#include "LayerBase.h"

namespace NEURAL_NETWORK 
{
	class LayerDropout : public LayerBase
	{
	public:
		LayerDropout(double rate);
		
		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		void SetDInput(const Eigen::MatrixXd& dinput) override { d_inputs_ = dinput; }
		Eigen::MatrixXd predictions() const override;
	
	private:
		double rate_;
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd mask_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYER_DROPOUT_H__