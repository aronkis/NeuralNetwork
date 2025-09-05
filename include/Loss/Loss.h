#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>
#include <memory>
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
						   const Eigen::MatrixXd& targets,
						   bool include_regularization = false);
		void CalculateAccumulatedLoss(bool include_regularization = false);
		void RegularizationLoss();
		void NewPass();
		void RememberTrainableLayers(const std::vector<std::weak_ptr<LayerDense>>& layers);

		const double GetRegularizationLoss() const;
		const double GetLoss() const;
		const double GetAccumulatedLoss() const;
        const Eigen::MatrixXd& GetOutput() const;
        const Eigen::MatrixXd& GetDInput() const;

		virtual void forward(const Eigen::MatrixXd& predictions, 
							 const Eigen::MatrixXd& targets) = 0;
		virtual void backward(const Eigen::MatrixXd& d_values,
							  const Eigen::MatrixXd& targets) = 0;

	protected:
		double loss_ = 0.0;

		Eigen::MatrixXd output_;

		Eigen::MatrixXd d_inputs_;

	private:
		std::vector<std::weak_ptr<LayerDense>> trainable_layers_;
		
		double regularization_loss_ = 0.0;
		double accumulated_loss_ = 0.0;
		int accumulated_count_ = 0;

	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_H__