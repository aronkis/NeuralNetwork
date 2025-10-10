#ifndef __LAYERDENSE_H__
#define __LAYERDENSE_H__

#include <Eigen/Dense>
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

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& d_values) override;
		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::MatrixXd& GetWeights() const;
		const Eigen::RowVectorXd& GetBiases() const;
		const Eigen::MatrixXd& GetDWeights() const;
		const Eigen::RowVectorXd& GetDBiases() const;

		const Eigen::MatrixXd& GetWeightMomentums() const;
		const Eigen::RowVectorXd& GetBiasMomentums() const;
		const Eigen::MatrixXd& GetWeightCaches() const;
		const Eigen::RowVectorXd& GetBiasCaches() const;

		double GetWeightRegularizerL1() const;
		double GetWeightRegularizerL2() const;
		double GetBiasRegularizerL1() const;
		double GetBiasRegularizerL2() const;

		std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const override;
		void SetParameters(const Eigen::MatrixXd& weights, 
						   const Eigen::RowVectorXd& biases) override;

		void SetDInput(const Eigen::MatrixXd& dinput) override;

		void SetWeightMomentums(const Eigen::MatrixXd& weight_momentums);
		void SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums);
		void SetWeightCaches(const Eigen::MatrixXd& weight_caches);
		void SetBiasCaches(const Eigen::RowVectorXd& bias_caches);

		void UpdateWeights(Eigen::MatrixXd& weight_update);
		void UpdateWeightsCache(Eigen::MatrixXd& weight_update);
		void UpdateBiases(Eigen::RowVectorXd& bias_update);
		void UpdateBiasesCache(Eigen::RowVectorXd& bias_update);


	private:
		Eigen::MatrixXd inputs_;
		Eigen::MatrixXd weights_;
		Eigen::RowVectorXd biases_;

		Eigen::MatrixXd output_;

		Eigen::MatrixXd d_inputs_;
		Eigen::MatrixXd d_weights_;
		Eigen::RowVectorXd d_biases_;

		Eigen::MatrixXd weight_momentums_;
		Eigen::RowVectorXd bias_momentums_;
		Eigen::MatrixXd weight_caches_;
		Eigen::RowVectorXd bias_caches_;

		double weight_regularizer_l1_;
		double weight_regularizer_l2_;
		double bias_regularizer_l1_;
		double bias_regularizer_l2_;
	};
} // namespace NEURAL_NETWORK

#endif // __LAYERDENSE_H__