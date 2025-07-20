#ifndef __LAYERDENSE_H__
#define __LAYERDENSE_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
	class LayerDense
	{
	public:
		LayerDense(int n_inputs, int n_neurons);
		~LayerDense() = default;

		LayerDense(const LayerDense&) = delete;
		LayerDense& operator=(const LayerDense&) = delete;

		void forward(const Eigen::MatrixXd& inputs);
		void backward(const Eigen::MatrixXd& d_values);

		const Eigen::MatrixXd& GetWeights() const;
		const Eigen::RowVectorXd& GetBiases() const;
		const Eigen::MatrixXd& GetOutput() const;
		
		const Eigen::MatrixXd& GetDInput() const;
		const Eigen::MatrixXd& GetDWeights() const;
		const Eigen::RowVectorXd& GetDBiases() const;

		const Eigen::MatrixXd& GetWeightMomentums() const;
		const Eigen::RowVectorXd& GetBiasMomentums() const;
		const Eigen::MatrixXd& GetWeightCaches() const;
		const Eigen::RowVectorXd& GetBiasCaches() const;

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
	};

} // namespace NEURAL_NETWORK

#endif // __LAYERDENSE_H__