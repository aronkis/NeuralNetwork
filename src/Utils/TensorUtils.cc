#include "TensorUtils.h"

void NEURAL_NETWORK::TensorUtils::MatrixToTensor4D(const Eigen::MatrixXd& matrix,
                                                    Eigen::Tensor<double, 4>& tensor,
                                                    int batch_size, int height,
                                                    int width, int channels)
{
    if (tensor.size() == 0 || tensor.dimension(0) != batch_size)
    {
        tensor = Eigen::Tensor<double, 4>(batch_size, height, width, channels);
    }

    auto expected_size = batch_size * height * width * channels;

    // Use explicit loop to ensure correct row-major ordering for spatial data
    {
        for (int b = 0; b < batch_size && b < matrix.rows(); b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        // Row-major spatial ordering: (H*W + W)*C + C
                        int matrix_idx = (h * width + w) * channels + c;
                        if (matrix_idx < matrix.cols())
                        {
                            tensor(b, h, w, c) = matrix(b, matrix_idx);
                        }
                    }
                }
            }
        }
    }
}

Eigen::MatrixXd NEURAL_NETWORK::TensorUtils::Tensor4DToMatrix(const Eigen::Tensor<double, 4>& tensor)
{
    int batch_size = tensor.dimension(0);
    int height = tensor.dimension(1);
    int width = tensor.dimension(2);
    int channels = tensor.dimension(3);
    auto expected_size = batch_size * height * width * channels;

    // Use explicit loop to ensure correct row-major ordering for spatial data
    {
        Eigen::MatrixXd matrix(batch_size, height * width * channels);

        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        // Row-major spatial ordering: (H*W + W)*C + C
                        int matrix_idx = (h * width + w) * channels + c;
                        if (matrix_idx < matrix.cols())
                        {
                            matrix(b, matrix_idx) = tensor(b, h, w, c);
                        }
                    }
                }
            }
        }

        return matrix;
    }
}

Eigen::MatrixXd NEURAL_NETWORK::TensorUtils::im2col(const Eigen::Tensor<double, 4> &input_tensor,
													int filter_height, int filter_width,
													int pad_h, int pad_w, int stride_h, int stride_w)
{
    int batch_size = input_tensor.dimension(0);
	int input_height = input_tensor.dimension(1);
	int input_width = input_tensor.dimension(2);
	int input_channels = input_tensor.dimension(3);

	int output_height = (input_height + 2 * pad_h - filter_height) / stride_h + 1;
	int output_width = (input_width + 2 * pad_w - filter_width) / stride_w + 1;

	int col_height = filter_height * filter_width * input_channels;
	int col_width = output_height * output_width * batch_size;

	Eigen::MatrixXd im2col_matrix(col_height, col_width);
	int col_index = 0;

	for (int b = 0; b < batch_size; b++)
	{
		for (int out_h = 0; out_h < output_height; out_h++)
		{
			for (int out_w = 0; out_w < output_width; out_w++)
			{
				int row_index = 0;
				for (int c = 0; c < input_channels; c++)
				{
					for (int fh = 0; fh < filter_height; fh++)
					{
						for (int fw = 0; fw < filter_width; fw++)
						{
							int h = out_h * stride_h - pad_h + fh;
							int w = out_w * stride_w - pad_w + fw;
							double value = 0.0;

							if (h >= 0 && h < input_height && w >= 0 && w < input_width)
							{
								value = input_tensor(b, h, w, c);
							}

							im2col_matrix(row_index, col_index) = value;
							row_index++;
						}
					}
				}
				col_index++;
			}
		}
	}

	return im2col_matrix;
}

void NEURAL_NETWORK::TensorUtils::col2im(const Eigen::MatrixXd &col_matrix,
                                         Eigen::Tensor<double, 4> &input_tensor,
					                     int batch_size, int input_height,
					                     int input_width, int input_channels,
					                     int filter_height, int filter_width,
					                     int pad_h, int pad_w,
					                     int stride_h, int stride_w)
{
   if (input_tensor.size() == 0 ||
		input_tensor.dimension(0) != batch_size ||
		input_tensor.dimension(1) != input_height ||
		input_tensor.dimension(2) != input_width ||
		input_tensor.dimension(3) != input_channels)
	{
		input_tensor = Eigen::Tensor<double, 4>(batch_size, input_height, input_width, input_channels);
	}

	input_tensor.setZero();

	int output_height = (input_height + 2 * pad_h - filter_height) / stride_h + 1;
	int output_width = (input_width + 2 * pad_w - filter_width) / stride_w + 1;

	int col_index = 0;

	for (int b = 0; b < batch_size; b++)
	{
		for (int out_h = 0; out_h < output_height; out_h++)
		{
			for (int out_w = 0; out_w < output_width; out_w++)
			{
				int row_index = 0;
				for (int c = 0; c < input_channels; c++)
				{
					for (int fh = 0; fh < filter_height; fh++)
					{
						for (int fw = 0; fw < filter_width; fw++)
						{
							int h = out_h * stride_h - pad_h + fh;
							int w = out_w * stride_w - pad_w + fw;
							double value = 0.0;

							if (h >= 0 && h < input_height && w >= 0 && w < input_width)
							{
								input_tensor(b, h, w, c) += col_matrix(row_index, col_index);
							}

							row_index++;
						}
					}
				}
				col_index++;
			}
		}
	} 
}                                         