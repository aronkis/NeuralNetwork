#ifndef __LOSS_MEAN_SQUARED_ERROR_H__
#define __LOSS_MEAN_SQUARED_ERROR_H__

#include "Loss.h"

namespace NEURAL_NETWORK 
{
	class LossMeanSquaredError : public Loss
	{
	public:
		LossMeanSquaredError() = default;
		~LossMeanSquaredError() = default;

		void forwardDouble(const Eigen::MatrixXd& predictions,
						   const Eigen::MatrixXd& targets);
		
		void backwardDouble(const Eigen::MatrixXd& d_values,
							const Eigen::MatrixXd& targets);

		void forward(const Eigen::MatrixXd& predictions,
					 const Eigen::MatrixXi& targets) override;

		void backward(const Eigen::MatrixXd& d_values,
					  const Eigen::MatrixXi& targets) override;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_MEAN_SQUARED_ERROR_H__
