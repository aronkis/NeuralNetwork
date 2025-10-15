#ifndef __ACCURACY_H__
#define __ACCURACY_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace NEURAL_NETWORK
{
	class Accuracy
	{
	public:
		Accuracy() = default;
		virtual ~Accuracy() = default;

		Accuracy(const Accuracy&) = delete;
		Accuracy& operator=(const Accuracy&) = delete;

		void Calculate(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& labels);
		void CalculateAccumulated();
		void NewPass();

		double GetAccuracy() const;
		double GetAccumulatedAccuracy() const;

		virtual void init(const Eigen::Tensor<double, 2>& target, bool reinit = false) = 0;

	protected:
		virtual Eigen::Tensor<double, 1> compare(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) const = 0;
	
	private:
		double accuracy_ = 0.0;
		double accumulated_accuracy_ = 0.0;
		int accumulated_count_ = 0;
	};
} // namespace NEURAL_NETWORK

#endif // __ACCURACY_H__