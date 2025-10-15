#ifndef __ACCURACY_CATEGORICAL_H__
#define __ACCURACY_CATEGORICAL_H__

#include "Accuracy.h"

namespace NEURAL_NETWORK
{
	class AccuracyCategorical : public Accuracy
	{
	public:
		AccuracyCategorical() = default;
		~AccuracyCategorical() = default;

		void init(const Eigen::Tensor<double, 2>& labels, bool reinit = false) override {}
		Eigen::Tensor<double, 1> compare(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) const override;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_CATEGORICAL_H__