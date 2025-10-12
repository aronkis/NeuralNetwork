#ifndef __TENSOR_UTILS_H__
#define __TENSOR_UTILS_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace NEURAL_NETWORK
{
    namespace TensorUtils
    {
        void MatrixToTensor4D(const Eigen::MatrixXd& matrix,
                              Eigen::Tensor<double, 4>& tensor,
                              int batch_size, int height, 
                              int width, int channels);

        Eigen::MatrixXd Tensor4DToMatrix(const Eigen::Tensor<double, 4>& tensor);

        Eigen::MatrixXd im2col(const Eigen::Tensor<double, 4> &input_tensor,
							   int filter_height, int filter_width,
							   int pad_h, int pad_w, int stride_h, int stride_w);

		void col2im(const Eigen::MatrixXd &col_matrix,
                    Eigen::Tensor<double, 4> &input_tensor,
					int batch_size, int input_height,
					int input_width, int input_channels,
					int filter_height, int filter_width,
					int pad_h, int pad_w,
					int stride_h, int stride_w);

    } // namespace TensorUtils
} // namespace NEURAL_NETWORK

#endif // __TENSOR_UTILS_H__
