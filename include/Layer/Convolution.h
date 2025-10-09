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
					int padding, int stride_height, int stride_width);

		~Convolution() = default;

		void forward(const Eigen::MatrixXd& inputs, bool training) override;
		void backward(const Eigen::MatrixXd& dvalues) override;
		const Eigen::MatrixXd& GetOutput() const override;
		const Eigen::MatrixXd& GetDInput() const override;
		Eigen::MatrixXd predictions() const override;

		const Eigen::Tensor<double, 4>& GetWeights() const;
		const Eigen::VectorXd& GetBiases() const;
		
		void SetDInput(const Eigen::MatrixXd& dinput) override;

	private:
		Eigen::Tensor<double, 4> inputs_;
		Eigen::Tensor<double, 4> weights_;
		Eigen::VectorXd biases_;

		Eigen::Tensor<double, 4> tensor_output_;
		Eigen::MatrixXd output_;

		Eigen::MatrixXd d_input_;
		Eigen::Tensor<double, 4> d_weights_;
		Eigen::VectorXd d_biases_;

		int input_height_ = -1;
		int input_width_ = -1;
		int padding_ = 0;
		int stride_height_ = 1;
		int stride_width_ = 1;


		Eigen::Tensor<double, 4> MatrixToTensor(const Eigen::MatrixXd& matrix,
												int batch_size, int height, 
												int width, int channels);
		Eigen::MatrixXd TensorToMatrix(const Eigen::Tensor<double, 4>& tensor);

		Eigen::MatrixXd WeightsToMatrix() const;
		Eigen::Tensor<double, 4> WeightsToTensor(const Eigen::MatrixXd& weights_matrix) const;

		Eigen::MatrixXd im2col(const Eigen::Tensor<double, 4> &input_tensor,
							   int filter_height, int filter_width,
							   int pad_h, int pad_w, int stride_h, int stride_w);

		Eigen::Tensor<double, 4> col2im(const Eigen::MatrixXd &col_matrix,
										int batch_size, int input_height, 
										int input_width, int input_channels,
										int filter_height, int filter_width,
										int pad_h, int pad_w, 
										int stride_h, int stride_w);
	};
} // namespace NEURAL_NETWORK

#endif // __CONVOLUTION_H__