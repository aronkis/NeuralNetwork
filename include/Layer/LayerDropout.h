#ifndef __LAYER_DROPOUT_H__
#define __LAYER_DROPOUT_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK 
{
	class LayerDropout
	{
	public:
		LayerDropout(double rate);
		
		void forward(const Eigen::MatrixXd& inputs);
		void backward(const Eigen::MatrixXd& dvalues);

		const Eigen::MatrixXd& GetOutput() const;
		const Eigen::MatrixXd& GetDInput() const;
	
	private:
		double rate_;
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd mask_;
		Eigen::MatrixXd output_;
		Eigen::MatrixXd d_inputs_;
	};
}
#endif // __LAYER_DROPOUT_H__