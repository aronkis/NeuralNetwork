#ifndef __LAYERDENSE_H__
#define __LAYERDENSE_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class LayerDense : public LayerBase
	{
	public:
		LayerDense(int n_inputs, int n_neurons,
				   double weight_regularizer_l1 = 0, double weight_regularizer_l2 = 0,
				   double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0);
		~LayerDense() = default;

		void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 2>& d_values) override;
		const Eigen::Tensor<double, 2>& GetOutput() const override;
		const Eigen::Tensor<double, 2>& GetDInput() const override;
		Eigen::Tensor<double, 2> predictions() const override;

		const Eigen::Tensor<double, 2>& GetWeights() const override;
		const Eigen::Tensor<double, 1>& GetBiases() const override;
		const Eigen::Tensor<double, 2>& GetDWeights() const override;
		const Eigen::Tensor<double, 1>& GetDBiases() const override;

		const Eigen::Tensor<double, 2>& GetWeightMomentums() const override;
		const Eigen::Tensor<double, 1>& GetBiasMomentums() const override;
		const Eigen::Tensor<double, 2>& GetWeightCaches() const override;
		const Eigen::Tensor<double, 1>& GetBiasCaches() const override;

		double GetWeightRegularizerL1() const override;
		double GetWeightRegularizerL2() const override;
		double GetBiasRegularizerL1() const override;
		double GetBiasRegularizerL2() const override;

		std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> GetParameters() const override;
		void SetParameters(const Eigen::Tensor<double, 2>& weights,
						   const Eigen::Tensor<double, 1>& biases) override;

		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;

		void SetWeightMomentums(const Eigen::Tensor<double, 2>& weight_momentums) override;
		void SetBiasMomentums(const Eigen::Tensor<double, 1>& bias_momentums) override;
		void SetWeightCaches(const Eigen::Tensor<double, 2>& weight_caches) override;
		void SetBiasCaches(const Eigen::Tensor<double, 1>& bias_caches) override;

		void UpdateWeights(Eigen::Tensor<double, 2>& weight_update) override;
		void UpdateWeightsCache(Eigen::Tensor<double, 2>& weight_update) override;
		void UpdateBiases(Eigen::Tensor<double, 1>& bias_update) override;
		void UpdateBiasesCache(Eigen::Tensor<double, 1>& bias_update) override;

	private:
		Eigen::Tensor<double, 2> inputs_;
		Eigen::Tensor<double, 2> weights_;
		Eigen::Tensor<double, 1> biases_;

		Eigen::Tensor<double, 2> output_;

		Eigen::Tensor<double, 2> d_inputs_;
		Eigen::Tensor<double, 2> d_weights_;
		Eigen::Tensor<double, 1> d_biases_;

		Eigen::Tensor<double, 2> weight_momentums_;
		Eigen::Tensor<double, 1> bias_momentums_;
		Eigen::Tensor<double, 2> weight_caches_;
		Eigen::Tensor<double, 1> bias_caches_;

		double weight_regularizer_l1_;
		double weight_regularizer_l2_;
		double bias_regularizer_l1_;
		double bias_regularizer_l2_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYERDENSE_H__