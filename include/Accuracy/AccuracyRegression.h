#ifndef __ACCURACY_REGRESSION_H__
#define __ACCURACY_REGRESSION_H__

#include "Accuracy.h"

namespace NEURAL_NETWORK
{
	class AccuracyRegression : public Accuracy
	{
	public:
		AccuracyRegression() = default;
		~AccuracyRegression() = default;

		void init(const Eigen::MatrixXd& target, bool reinit = false) override;
		Eigen::ArrayXd compare(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& targets) const override;
	
	private:
		double epsilon_ = -1.0;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_REGRESSION_H__