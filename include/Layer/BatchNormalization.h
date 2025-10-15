#ifndef __BATCH_NORMALIZATION_H__
#define __BATCH_NORMALIZATION_H__

#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class BatchNormalization : public LayerBase
	{
	public:
		BatchNormalization(int num_features,
						   double epsilon = 1e-5,
						   double momentum = 0.1);
		~BatchNormalization() = default;

		void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 2>& dvalues) override;

		const Eigen::Tensor<double, 2>& GetOutput() const override;
		const Eigen::Tensor<double, 2>& GetDInput() const override;
		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;

		const Eigen::Tensor<double, 2>& GetWeights() const override;
		const Eigen::Tensor<double, 1>& GetBiases() const override;
		const Eigen::Tensor<double, 2>& GetDWeights() const override;
		const Eigen::Tensor<double, 1>& GetDBiases() const override;

		void UpdateWeights(Eigen::Tensor<double, 2>& weight_update) override;
		void UpdateBiases(Eigen::Tensor<double, 1>& bias_update) override;

		// Parameter management for model serialization
		std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> GetParameters() const override;
		void SetParameters(const Eigen::Tensor<double, 2>& weights, const Eigen::Tensor<double, 1>& biases) override;

		// Getter for num_features (needed for model serialization)
		int GetNumFeatures() const;

	private:
		Eigen::Tensor<double, 2> gamma_;
		Eigen::Tensor<double, 1> beta_;

		Eigen::Tensor<double, 2> d_gamma_;
		Eigen::Tensor<double, 1> d_beta_;
		Eigen::Tensor<double, 2> d_input_;

		Eigen::Tensor<double, 1> running_mean_;
		Eigen::Tensor<double, 1> running_var_;

		Eigen::Tensor<double, 2> cached_input_;
		Eigen::Tensor<double, 1> cached_mean_;
		Eigen::Tensor<double, 1> cached_var_;
		Eigen::Tensor<double, 2> cached_normalized_;

		Eigen::Tensor<double, 2> output_;

		int num_features_;
		int batch_size_;
		double epsilon_;
		double momentum_;
	};

} // namespace NEURAL_NETWORK

#endif // __BATCH_NORMALIZATION_H__