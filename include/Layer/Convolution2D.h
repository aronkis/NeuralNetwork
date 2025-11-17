#ifndef __CONVOLUTION_2D_H__
#define __CONVOLUTION_2D_H__

// #define EIGEN_USE_BLAS -- to be tested
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class Convolution2D : public LayerBase
	{
	public:
		Convolution2D(int number_of_filters, int filter_height, int filter_width,
					int input_height, int input_width, int input_channels,
					int padding, int stride_height, int stride_width,
					double weight_regularizer_l1 = 0.0, double weight_regularizer_l2 = 0.0,
					double bias_regularizer_l1 = 0.0, double bias_regularizer_l2 = 0.0);

		~Convolution2D() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::Tensor<double, 4>& GetWeightsTensor() const;
		const Eigen::VectorXd& GetBiasesVector() const;

		int GetNumberOfFilters() const;
		int GetFilterHeight() const;
		int GetFilterWidth() const;
		int GetInputHeight() const;
		int GetInputWidth() const;
		int GetInputChannels() const;
		int GetPadding() const;
		int GetStrideHeight() const;
		int GetStrideWidth() const;

		double GetWeightRegularizerL1() const override;
		double GetWeightRegularizerL2() const override;
		double GetBiasRegularizerL1() const override;
		double GetBiasRegularizerL2() const override;
		const Eigen::MatrixXd& GetWeights() const override;
		const Eigen::RowVectorXd& GetBiases() const override;

		std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const override;
		void SetParameters(const Eigen::MatrixXd& weights, 
						   const Eigen::RowVectorXd& biases) override;

		void SetDInput(const Eigen::MatrixXd& dinput) override;

		const Eigen::MatrixXd& GetDWeights() const override;
		const Eigen::RowVectorXd& GetDBiases() const override;

		const Eigen::MatrixXd& GetWeightMomentums() const override;
		const Eigen::RowVectorXd& GetBiasMomentums() const override;
		const Eigen::MatrixXd& GetWeightCaches() const override;
		const Eigen::RowVectorXd& GetBiasCaches() const override;

		void SetWeightMomentums(const Eigen::MatrixXd& weight_momentums) override;
		void SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums) override;
		void SetWeightCaches(const Eigen::MatrixXd& weight_caches) override;
		void SetBiasCaches(const Eigen::RowVectorXd& bias_caches) override;

		void UpdateWeights(Eigen::MatrixXd& weight_update) override;
		void UpdateWeightsCache(Eigen::MatrixXd& weight_update) override;
		void UpdateBiases(Eigen::RowVectorXd& bias_update) override;
		void UpdateBiasesCache(Eigen::RowVectorXd& bias_update) override;

	private:
		Eigen::Tensor<double, 4> inputs_;
		Eigen::Tensor<double, 4> weights_;
		Eigen::VectorXd biases_;

		Eigen::Tensor<double, 4> tensor_output_;
		Eigen::MatrixXd output_;

		Eigen::MatrixXd d_input_;
		Eigen::Tensor<double, 4> d_input_tensor_;
		Eigen::Tensor<double, 4> d_weights_;
		Eigen::VectorXd d_biases_;

		Eigen::Tensor<double, 4> weight_momentums_;
		Eigen::VectorXd bias_momentums_;
		Eigen::Tensor<double, 4> weight_caches_;
		Eigen::VectorXd bias_caches_;

		Eigen::MatrixXd im2col_input_;

		int input_height_ = -1;
		int input_width_ = -1;
		int input_channels_ = -1;
		int filter_height_ = -1;
		int filter_width_ = -1;
		int number_of_filters_ = -1;
		int padding_ = 0;
		int stride_height_ = 1;
		int stride_width_ = 1;
		int pad_height_;
		int pad_width_;

		double weight_regularizer_l1_;
		double weight_regularizer_l2_;
		double bias_regularizer_l1_;
		double bias_regularizer_l2_;

		mutable Eigen::MatrixXd weights_matrix_cache_;
		mutable Eigen::RowVectorXd biases_cache_;
		mutable Eigen::MatrixXd d_weights_matrix_cache_;
		mutable Eigen::RowVectorXd d_biases_cache_;
		mutable Eigen::MatrixXd weight_momentums_matrix_cache_;
		mutable Eigen::RowVectorXd bias_momentums_cache_;
		mutable Eigen::MatrixXd weight_caches_matrix_cache_;
		mutable Eigen::RowVectorXd bias_caches_cache_;


		void InputMatrixToTensor(const Eigen::MatrixXd& matrix,
								 int batch_size, int height,
								 int width, int channels);
		Eigen::MatrixXd InputTensorToMatrix(const Eigen::Tensor<double, 4>& tensor);

		void WeightsToTensor(const Eigen::MatrixXd& weights_matrix);
		Eigen::MatrixXd WeightsToMatrix() const;

		Eigen::MatrixXd im2col(const Eigen::Tensor<double, 4> &input_tensor,
							   int filter_height, int filter_width,
							   int pad_h, int pad_w, int stride_h, int stride_w);

		void col2im(const Eigen::MatrixXd &col_matrix,
					int batch_size, int input_height,
					int input_width, int input_channels,
					int filter_height, int filter_width,
					int pad_h, int pad_w,
					int stride_h, int stride_w);

		// Helper methods for tensor-matrix conversions
		Eigen::MatrixXd WeightsToMatrixFromTensor(const Eigen::Tensor<double, 4>& tensor) const;
		void MatrixToWeightsTensor(const Eigen::MatrixXd& matrix, 
								   Eigen::Tensor<double, 4>& tensor) const;

		// Tensor-specific methods for internal use
		void SetWeightMomentumsTensor(const Eigen::Tensor<double, 4>& weight_momentums);
		void SetBiasMomentumsTensor(const Eigen::VectorXd& bias_momentums);
		void SetWeightCachesTensor(const Eigen::Tensor<double, 4>& weight_caches);
		void SetBiasCachesTensor(const Eigen::VectorXd& bias_caches);

		void UpdateWeightsTensor(Eigen::Tensor<double, 4>& weight_update);
		void UpdateWeightsCacheTensor(Eigen::Tensor<double, 4>& weight_update);
		void UpdateBiasesTensor(Eigen::VectorXd& bias_update);
		void UpdateBiasesCacheTensor(Eigen::VectorXd& bias_update);
	};
} // namespace NEURAL_NETWORK

#endif // __CONVOLUTION_2D_H__