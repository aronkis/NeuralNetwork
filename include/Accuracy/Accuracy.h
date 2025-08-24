#ifndef __ACCURACY_H__
#define __ACCURACY_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK 
{
	class Accuracy
	{
	public:
		Accuracy() = default;
		virtual ~Accuracy() = default;

		Accuracy(const Accuracy&) = delete;
		Accuracy& operator=(const Accuracy&) = delete;

		double Calculate(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& labels);
		virtual void init(const Eigen::MatrixXd& target, bool reinit = false) = 0;

	protected:
		virtual Eigen::ArrayXd compare(const Eigen::MatrixXd& predictions, Eigen::MatrixXd& targets) const = 0;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_H__