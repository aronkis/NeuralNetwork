#ifndef __LOSS_MEAN_ABSOLUTE_ERROR_H__
#define __LOSS_MEAN_ABSOLUTE_ERROR_H__

#include "Loss.h"

namespace NEURAL_NETWORK 
{
	class LossMeanAbsoluteError : public Loss
	{
	public:
		LossMeanAbsoluteError() = default;
		~LossMeanAbsoluteError() = default;

		void forward(const Eigen::MatrixXd& predictions,
					 const Eigen::MatrixXd& targets) override;

		void backward(const Eigen::MatrixXd& d_values,
					  const Eigen::MatrixXd& targets) override;
	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_MEAN_ABSOLUTE_ERROR_H__