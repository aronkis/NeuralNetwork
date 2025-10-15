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

		void forward(const Eigen::Tensor<double, 2>& predictions,
					 const Eigen::Tensor<double, 2>& targets) override;

		void backward(const Eigen::Tensor<double, 2>& d_values,
					  const Eigen::Tensor<double, 2>& targets) override;
	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_CATEGORICAL_CROSSENTROPY_H__