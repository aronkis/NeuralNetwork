#ifndef __LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "Loss.h"

namespace NEURAL_NETWORK
{
	class LossCategoricalCrossEntropy : public Loss
	{
	public:
		LossCategoricalCrossEntropy() = default;
		~LossCategoricalCrossEntropy() = default;

		void forward(const Eigen::MatrixXd& predictions,
					 const Eigen::MatrixXd& targets) override;

		void backward(const Eigen::MatrixXd& d_values,
					  const Eigen::MatrixXd& targets) override;
	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_CATEGORICAL_CROSSENTROPY_H__