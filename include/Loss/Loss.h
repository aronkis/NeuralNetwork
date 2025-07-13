#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{

	class Loss
	{
	public:
		Loss() = default;
		virtual ~Loss() = default;

		Loss(const Loss&) = delete;
		Loss& operator=(const Loss&) = delete;

		void calculateLoss(const Eigen::MatrixXd& predictions,
						   const Eigen::MatrixXi& targets);

		double GetLoss() const;

	protected:
		virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& predictions,
										const Eigen::MatrixXi& targets) = 0;

		virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& d_values,
										 const Eigen::MatrixXi& targets) = 0;

	private:
		double loss_ = 0.0;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_H__