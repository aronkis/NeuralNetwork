#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class Loss
	{
	public:
		Loss() = default;
		virtual ~Loss() = default;

		Loss(const Loss&) = delete;
		Loss& operator=(const Loss&) = delete;

		void CalculateLoss(const Eigen::Tensor<double, 2>& predictions,
						   const Eigen::Tensor<double, 2>& targets,
						   bool include_regularization = false);
		void CalculateAccumulatedLoss(bool include_regularization = false);
		void RegularizationLoss();
		void NewPass();
		void RememberTrainableLayers(const std::vector<std::weak_ptr<LayerBase>>& layers);

		const double GetRegularizationLoss() const;
		const double GetLoss() const;
		const double GetAccumulatedLoss() const;
        const Eigen::Tensor<double, 2>& GetOutput() const;
        const Eigen::Tensor<double, 2>& GetDInput() const;

		virtual void forward(const Eigen::Tensor<double, 2>& predictions,
							 const Eigen::Tensor<double, 2>& targets) = 0;
		virtual void backward(const Eigen::Tensor<double, 2>& d_values,
							  const Eigen::Tensor<double, 2>& targets) = 0;

	protected:
		double loss_ = 0.0;

		Eigen::Tensor<double, 2> output_;

		Eigen::Tensor<double, 2> d_inputs_;

	private:
		std::vector<std::weak_ptr<LayerBase>> trainable_layers_;
		
		double regularization_loss_ = 0.0;
		double accumulated_loss_ = 0.0;
		int accumulated_count_ = 0;

	};
} // namespace NEURAL_NETWORK

#endif // __LOSS_H__