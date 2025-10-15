#ifndef __ADAM_H__
#define __ADAM_H__

#include "Optimizer.h"

namespace NEURAL_NETWORK { class BatchNormalization; }

namespace NEURAL_NETWORK
{
	class Adam : public Optimizer
	{
	public:
		Adam(double learning_rate = 0.001,
			 double decay = 0.0,
			 double beta1 = 0.9,
			 double beta2 = 0.999,
			 double epsilon = 1e-7);
		~Adam() = default;

		void UpdateParameters(NEURAL_NETWORK::LayerBase& layer) override;

		double GetBeta1() const;
		double GetBeta2() const;
		double GetEpsilon() const;

	private:
		void UpdateBatchNormalizationParameters(NEURAL_NETWORK::BatchNormalization& bn_layer);

		double beta1_;
		double beta2_;
		double epsilon_;

		// BatchNormalization specific momentum and cache storage
		Eigen::Tensor<double, 2> bn_gamma_momentum_;
		Eigen::Tensor<double, 1> bn_beta_momentum_;
		Eigen::Tensor<double, 2> bn_gamma_cache_;
		Eigen::Tensor<double, 1> bn_beta_cache_;
	};
} // namespace NEURAL_NETWORK

#endif // __ADAM_H__