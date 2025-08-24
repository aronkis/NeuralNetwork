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

		void init(const Eigen::MatrixXd& labels, bool reinit = false) override {}
		Eigen::ArrayXd compare(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& targets) const override;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_CATEGORICAL_H__