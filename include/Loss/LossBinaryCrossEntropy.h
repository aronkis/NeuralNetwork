#ifndef __LOSS_BINARY_CROSS_ENTROPY_H__
#define __LOSS_BINARY_CROSS_ENTROPY_H__

#include "Loss.h"

namespace NEURAL_NETWORK
{
	class LossBinaryCrossEntropy : public Loss
	{
	public:
		LossBinaryCrossEntropy() = default;
		~LossBinaryCrossEntropy() = default;

		void forward(const Eigen::Tensor<double, 2>& predictions,
					 const Eigen::Tensor<double, 2>& targets) override;

		void backward(const Eigen::Tensor<double, 2>& d_values,
					  const Eigen::Tensor<double, 2>& targets) override;
	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_BINARY_CROSS_ENTROPY_H__