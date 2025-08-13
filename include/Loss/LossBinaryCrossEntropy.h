#ifndef __LOSS_BINARY_CROSS_ENTROPY_H__
#define __LOSS_BINARY_CROSS_ENTROPY_H__

#include "Loss/Loss.h"

namespace NEURAL_NETWORK
{
	class LossBinaryCrossEntropy : public Loss
	{
	public:
		LossBinaryCrossEntropy() = default;
		~LossBinaryCrossEntropy() = default;

		LossBinaryCrossEntropy(const LossBinaryCrossEntropy&) = delete;
		LossBinaryCrossEntropy& operator=(const LossBinaryCrossEntropy&) = delete;

		void forward(const Eigen::MatrixXd& predictions,
					 const Eigen::MatrixXi& targets) override;

		void backward(const Eigen::MatrixXd& d_values,
					  const Eigen::MatrixXi& targets) override;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_BINARY_CROSS_ENTROPY_H__