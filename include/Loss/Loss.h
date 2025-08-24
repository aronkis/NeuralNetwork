#ifndef __LOSS_H__
#define __LOSS_H__

#include "LayerDense.h"
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

		void CalculateLoss(const Eigen::MatrixXd& predictions,
						   const Eigen::MatrixXi& targets,
						   bool include_regularization = false);

		void RememberTrainableLayers(const std::vector<LayerDense*>& layers);

		double RegularizationLoss() const;

		double GetRegularizationLoss() const;
		double GetLoss() const;
		
        const Eigen::MatrixXd& GetOutput() const;
        const Eigen::MatrixXd& GetDInput() const;

		virtual void forward(const Eigen::MatrixXd& predictions, 
							 const Eigen::MatrixXi& targets) = 0;
		virtual void backward(const Eigen::MatrixXd& d_values,
							  const Eigen::MatrixXi& targets) = 0;
		
		protected:
			double loss_ = 0.0;
			double regularization_loss_ = 0.0;

			std::vector<LayerDense*> trainable_layers_;

			Eigen::MatrixXd output_;

			Eigen::MatrixXd d_inputs_;
	};

} // namespace NEURAL_NETWORK

#endif // __LOSS_H__