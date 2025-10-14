#ifndef __BATCH_NORMALIZATION_H__
#define __BATCH_NORMALIZATION_H__

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

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;

		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		void SetDInput(const Eigen::MatrixXd& dinput) override;

		const Eigen::MatrixXd& GetWeights() const override;
		const Eigen::RowVectorXd& GetBiases() const override;
		const Eigen::MatrixXd& GetDWeights() const override;
		const Eigen::RowVectorXd& GetDBiases() const override;

		void UpdateWeights(Eigen::MatrixXd& weight_update) override;
		void UpdateBiases(Eigen::RowVectorXd& bias_update) override;

		// Parameter management for model serialization
		std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const override;
		void SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases) override;

		// Getter for num_features (needed for model serialization)
		int GetNumFeatures() const;

	private:
		Eigen::MatrixXd gamma_;
		Eigen::RowVectorXd beta_;

		Eigen::MatrixXd d_gamma_;
		Eigen::RowVectorXd d_beta_;
		Eigen::MatrixXd d_input_;

		Eigen::VectorXd running_mean_;
		Eigen::VectorXd running_var_;

		Eigen::MatrixXd cached_input_;
		Eigen::VectorXd cached_mean_;
		Eigen::VectorXd cached_var_;
		Eigen::MatrixXd cached_normalized_;

		Eigen::MatrixXd output_;

		int num_features_;
		int batch_size_;
		double epsilon_;
		double momentum_;
	};

} // namespace NEURAL_NETWORK

#endif // __BATCH_NORMALIZATION_H__