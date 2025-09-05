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

		void Calculate(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& labels);
		void CalculateAccumulated();
		void NewPass();

		double GetAccuracy() const;
		double GetAccumulatedAccuracy() const;

		virtual void init(const Eigen::MatrixXd& target, bool reinit = false) = 0;

	protected:
		virtual Eigen::ArrayXd compare(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) const = 0;
	
	private:
		double accuracy_ = 0.0;
		double accumulated_accuracy_ = 0.0;
		int accumulated_count_ = 0;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_H__