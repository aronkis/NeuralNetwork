#ifndef __LOSS_MEAN_ABSOLUTE_ERROR_H__
#define __LOSS_MEAN_ABSOLUTE_ERROR_H__

#include <Eigen/Dense>
#include "Loss.h"

namespace NEURAL_NETWORK {

class LossMeanAbsoluteError : public Loss
{
public:
	LossMeanAbsoluteError() = default;
	~LossMeanAbsoluteError() = default;

	LossMeanAbsoluteError(const LossMeanAbsoluteError&) = delete;
	LossMeanAbsoluteError& operator=(const LossMeanAbsoluteError&) = delete;

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

#endif // __LOSS_MEAN_ABSOLUTE_ERROR_H__