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
		~LayerDropout() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		double GetRate() const;
		
		void SetDInput(const Eigen::MatrixXd& dinput) override;
	
	private:
		Eigen::MatrixXd inputs_;
		
		Eigen::MatrixXd output_;
		
		Eigen::MatrixXd d_inputs_;

		Eigen::MatrixXd mask_;
		
		double rate_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYER_DROPOUT_H__