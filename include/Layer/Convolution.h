#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

// #define EIGEN_USE_BLAS -- to be tested
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
	class Convolution : public LayerBase
	{
	public:
		Convolution(int number_of_filters, int filter_height, int filter_width,
					int input_height, int input_width, int input_channels,
					int padding, int stride_height, int stride_width,
					double weight_regularizer_l1 = 0.0, double weight_regularizer_l2 = 0.0,
					double bias_regularizer_l1 = 0.0, double bias_regularizer_l2 = 0.0);

		~Convolution() = default;

		void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
		void backward(const Eigen::Tensor<double, 2>& dvalues) override;
		const Eigen::Tensor<double, 2>& GetOutput() const override;
		const Eigen::Tensor<double, 2>& GetDInput() const override;
		Eigen::Tensor<double, 2> predictions() const override;

		const Eigen::Tensor<double, 4>& GetWeightsTensor() const;
		const Eigen::Tensor<double, 1>& GetBiasesVector() const;

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
		const Eigen::Tensor<double, 2>& GetWeights() const override;
		const Eigen::Tensor<double, 1>& GetBiases() const override;

		std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> GetParameters() const override;
		void SetParameters(const Eigen::Tensor<double, 2>& weights,
						   const Eigen::Tensor<double, 1>& biases) override;

		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;

		// LayerBase virtual method overrides for gradients
		const Eigen::Tensor<double, 2>& GetDWeights() const override;
		const Eigen::Tensor<double, 1>& GetDBiases() const override;

		// LayerBase virtual method overrides for momentums/caches (Tensor interface)
		const Eigen::Tensor<double, 2>& GetWeightMomentums() const override;
		const Eigen::Tensor<double, 1>& GetBiasMomentums() const override;
		const Eigen::Tensor<double, 2>& GetWeightCaches() const override;
		const Eigen::Tensor<double, 1>& GetBiasCaches() const override;

		void SetWeightMomentums(const Eigen::Tensor<double, 2>& weight_momentums) override;
		void SetBiasMomentums(const Eigen::Tensor<double, 1>& bias_momentums) override;
		void SetWeightCaches(const Eigen::Tensor<double, 2>& weight_caches) override;
		void SetBiasCaches(const Eigen::Tensor<double, 1>& bias_caches) override;

		void UpdateWeights(Eigen::Tensor<double, 2>& weight_update) override;
		void UpdateWeightsCache(Eigen::Tensor<double, 2>& weight_update) override;
		void UpdateBiases(Eigen::Tensor<double, 1>& bias_update) override;
		void UpdateBiasesCache(Eigen::Tensor<double, 1>& bias_update) override;

	private:
		Eigen::Tensor<double, 4> inputs_;
		Eigen::Tensor<double, 4> weights_;
		Eigen::Tensor<double, 1> biases_;

		Eigen::Tensor<double, 4> tensor_output_;
		Eigen::Tensor<double, 2> output_;

		Eigen::Tensor<double, 2> d_input_;
		Eigen::Tensor<double, 4> d_input_tensor_;
		Eigen::Tensor<double, 4> d_weights_;
		Eigen::Tensor<double, 1> d_biases_;

		Eigen::Tensor<double, 4> weight_momentums_;
		Eigen::Tensor<double, 1> bias_momentums_;
		Eigen::Tensor<double, 4> weight_caches_;
		Eigen::Tensor<double, 1> bias_caches_;

		Eigen::Tensor<double, 2> im2col_input_;

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

		mutable Eigen::Tensor<double, 2> weights_matrix_cache_;
		mutable Eigen::Tensor<double, 1> biases_cache_;
		mutable Eigen::Tensor<double, 2> d_weights_matrix_cache_;
		mutable Eigen::Tensor<double, 1> d_biases_cache_;
		mutable Eigen::Tensor<double, 2> weight_momentums_matrix_cache_;
		mutable Eigen::Tensor<double, 1> bias_momentums_cache_;
		mutable Eigen::Tensor<double, 2> weight_caches_matrix_cache_;
		mutable Eigen::Tensor<double, 1> bias_caches_cache_;


		void InputTensorToTensor(const Eigen::Tensor<double, 2>& tensor2d,
							int batch_size, int height,
							int width, int channels);
		Eigen::Tensor<double, 2> InputTensorToTensor2D(const Eigen::Tensor<double, 4>& tensor);

		void WeightsToTensor(const Eigen::Tensor<double, 2>& weights_tensor2d);
		Eigen::Tensor<double, 2> WeightsToTensor2D() const;

		Eigen::Tensor<double, 2> im2col(const Eigen::Tensor<double, 4> &input_tensor,
							   int filter_height, int filter_width,
							   int pad_h, int pad_w, int stride_h, int stride_w);

		void col2im(const Eigen::Tensor<double, 2> &col_tensor,
					int batch_size, int input_height,
					int input_width, int input_channels,
					int filter_height, int filter_width,
					int pad_h, int pad_w,
					int stride_h, int stride_w);

		// Helper methods for tensor conversions
		Eigen::Tensor<double, 2> WeightsTensor2DFromTensor4D(const Eigen::Tensor<double, 4>& tensor) const;
		void Tensor2DToWeightsTensor4D(const Eigen::Tensor<double, 2>& tensor2d, Eigen::Tensor<double, 4>& tensor4d) const;

		// Tensor-specific methods for internal use
		void SetWeightMomentumsTensor(const Eigen::Tensor<double, 4>& weight_momentums);
		void SetBiasMomentumsTensor(const Eigen::Tensor<double, 1>& bias_momentums);
		void SetWeightCachesTensor(const Eigen::Tensor<double, 4>& weight_caches);
		void SetBiasCachesTensor(const Eigen::Tensor<double, 1>& bias_caches);

		void UpdateWeightsTensor(Eigen::Tensor<double, 4>& weight_update);
		void UpdateWeightsCacheTensor(Eigen::Tensor<double, 4>& weight_update);
		void UpdateBiasesTensor(Eigen::Tensor<double, 1>& bias_update);
		void UpdateBiasesCacheTensor(Eigen::Tensor<double, 1>& bias_update);
	};
} // namespace NEURAL_NETWORK

#endif // __CONVOLUTION_H__