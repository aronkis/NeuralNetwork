#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>
#include "LayerDense.h"

namespace NEURAL_NETWORK
{
	class Loss
	{
	public:
		Loss() = default;
		virtual ~Loss() = default;

		Loss(const Loss&) = delete;
		Loss& operator=(const Loss&) = delete;

		void CalculateLoss(const Eigen::MatrixXd& predictions,
						   const Eigen::MatrixXi& targets);

		double RegularizationLoss(LayerDense& layer) const;

		double GetLoss() const;

        const Eigen::MatrixXd& GetOutput() const;
        const Eigen::MatrixXd& GetDInput() const;

	protected:
		virtual void forward(const Eigen::MatrixXd& predictions,
							 const Eigen::MatrixXi& targets) = 0;

		virtual void backward(const Eigen::MatrixXd& d_values,
							  const Eigen::MatrixXi& targets) = 0;

		double loss_ = 0.0;
		
		Eigen::MatrixXd output_;

        Eigen::MatrixXd d_inputs_;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_H__