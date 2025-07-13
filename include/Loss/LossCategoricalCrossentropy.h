#ifndef __LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "Loss/Loss.h"

namespace NEURAL_NETWORK
{
	class LossCategoricalCrossentropy : public Loss
	{
	public:
		LossCategoricalCrossentropy() = default;
		~LossCategoricalCrossentropy() = default;

		LossCategoricalCrossentropy(const LossCategoricalCrossentropy&) = delete;
		LossCategoricalCrossentropy& operator=(const LossCategoricalCrossentropy&) = delete;

		Eigen::MatrixXd forward(const Eigen::MatrixXd& predictions,
								const Eigen::MatrixXi& targets) override;

		Eigen::MatrixXd backward(const Eigen::MatrixXd& d_values,
										const Eigen::MatrixXi& targets) override;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_CATEGORICAL_CROSSENTROPY_H__