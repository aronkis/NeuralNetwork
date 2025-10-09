#include <random>
#include <cmath>

#include "Convolution.h"

NEURAL_NETWORK::Convolution::Convolution(int number_of_filters, int filter_height, int filter_width,
										 int input_height, int input_width, int input_channels,  
										 int padding, int stride_height, int stride_width)
{
	std::mt19937 gen(0);
	// He normal initialization
	const int fan_in = filter_height * filter_width * input_channels;
	const double he_std = std::sqrt(2.0 / static_cast<double>(fan_in));
	std::normal_distribution<> he_dist(0.0, he_std);
	weights_ = Eigen::Tensor<double, 4>(filter_height, filter_width, input_channels, number_of_filters);
	weights_ = weights_.unaryExpr([&](double) { return he_dist(gen); });

	biases_ = Eigen::VectorXd::Zero(number_of_filters);

	input_height_ = input_height;
	input_width_ = input_width;
	padding_ = padding;
	stride_height_ = stride_height;
	stride_width_ = stride_width;
}

void NEURAL_NETWORK::Convolution::forward(const Eigen::MatrixXd& inputs, bool training)
{
	int batch_size = inputs.rows();
	int input_features = inputs.cols();
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);
	int pad_height = padding_ ? (filter_height - 1) / 2 : 0;
  	int pad_width = padding_ ? (filter_width - 1) / 2 : 0;

	inputs_ = MatrixToTensor(inputs, batch_size,
							 input_height_,
							 input_width_,
							 input_channels);

	Eigen::MatrixXd im2col_matrix = im2col(inputs_,
										   filter_height, filter_width,
										   pad_height, pad_width,
										   stride_height_, stride_width_);

	const Eigen::MatrixXd filter_matrix = WeightsToMatrix();

	Eigen::MatrixXd result = filter_matrix * im2col_matrix; // Convolution

	result.colwise() += biases_; 

	int output_height = ((input_height_ + (2 * pad_height) - filter_height) /
						  stride_height_) + 1;
	int output_width =  ((input_width_ + (2 * pad_width) - filter_width) /
						  stride_width_) + 1;

	tensor_output_ = Eigen::Tensor<double, 4>(batch_size, output_height, 
											  output_width, num_filters);

	Eigen::MatrixXd result_transposed = result.transpose();

	int total_spatial = output_height * output_width * batch_size;
	Eigen::Map<Eigen::MatrixXd> tensor_map(tensor_output_.data(), 
										   total_spatial, num_filters);

	tensor_map = result_transposed;

	output_ = TensorToMatrix(tensor_output_);
}

void NEURAL_NETWORK::Convolution::backward(const Eigen::MatrixXd &d_values)
{
	int batch_size = d_values.rows();
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	int pad_height = padding_ ? (filter_height - 1) / 2 : 0;
	int pad_width = padding_ ? (filter_width - 1) / 2 : 0;

	int output_height = ((input_height_ + (2 * pad_height) - filter_height) /
						 stride_height_) + 1;
	int output_width = ((input_width_ + (2 * pad_width) - filter_width) /
						stride_width_) + 1;

	int total_spatial = batch_size * output_height * output_width;
	Eigen::Map<const Eigen::MatrixXd> d_values_reshaped(d_values.data(),
														total_spatial, num_filters);

	d_biases_ = d_values_reshaped.colwise().sum().transpose();

	Eigen::MatrixXd im2col_input = im2col(inputs_,
										  filter_height, filter_width,
										  pad_height, pad_width,
										  stride_height_, stride_width_); 

	Eigen::MatrixXd d_weights_matrix = im2col_input * d_values_reshaped;
	
	d_weights_ = WeightsToTensor(d_weights_matrix);

	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::MatrixXd d_input_col = weights_matrix.transpose() * d_values_reshaped.transpose();

	Eigen::Tensor<double, 4> d_input_tensor = col2im(d_input_col, d_values.rows(), input_height_,
													 input_width_, input_channels, filter_height,
													 filter_width, pad_height, pad_width,
													 stride_height_, stride_width_);
	d_input_ = TensorToMatrix(d_input_tensor);
}

//maybe weights in the n-th filter? GetWeight(int n)
const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Convolution::GetWeights() const 
{
	return weights_;
}

const Eigen::VectorXd& NEURAL_NETWORK::Convolution::GetBiases() const
{
	return biases_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution::GetDInput() const
{
	return d_input_;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::predictions() const
{
	return output_;
}


void NEURAL_NETWORK::Convolution::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_input_ = dinput;
}

Eigen::Tensor<double, 4> NEURAL_NETWORK::Convolution::MatrixToTensor(const Eigen::MatrixXd& matrix,
														int batch_size, int height, int width, int channels)
{
	Eigen::Tensor<double, 4> tensor(batch_size, height, width, channels);

	for (int b = 0; b < batch_size; b++)
	{
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int c = 0; c < channels; c++)
				{
					int matrix_idx = h * width * channels + w * channels + c;
					if (matrix_idx < matrix.cols())
					{
						tensor(b, h, w, c) = matrix(b, matrix_idx);
					}
				}
			}
		}
	}

	return tensor;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::TensorToMatrix(const Eigen::Tensor<double, 4>& tensor)
{
	int batch_size = tensor.dimension(0);
	int height = tensor.dimension(1);
	int width = tensor.dimension(2);
	int channels = tensor.dimension(3);

	Eigen::MatrixXd matrix(batch_size, height * width * channels);

	for (int b = 0; b < batch_size; b++)
	{
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int c = 0; c < channels; c++)
				{
					int matrix_idx = h * width * channels + w * channels + c;
					matrix(b, matrix_idx) = tensor(b, h, w, c);
				}
			}
		}
	}

	return matrix;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::WeightsToMatrix() const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	Eigen::MatrixXd weights_matrix(num_filters, filter_height * filter_width * input_channels);

	for (int f = 0; f < num_filters; f++)
	{
		int col = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					weights_matrix(f, col++) = weights_(h, w, c, f);
				}
			}
		}
	}

	return weights_matrix;
}

Eigen::Tensor<double, 4> NEURAL_NETWORK::Convolution::WeightsToTensor(const Eigen::MatrixXd& weights_matrix) const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	Eigen::Tensor<double, 4> weights_tensor(filter_height, filter_width, input_channels, num_filters);

	for (int f = 0; f < num_filters; f++)
	{
		int row = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					weights_tensor(h, w, c, f) = weights_matrix(row++, f);
				}
			}
		}
	}

	return weights_tensor;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::im2col(const Eigen::Tensor<double, 4> &input_tensor,
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

Eigen::Tensor<double, 4> NEURAL_NETWORK::Convolution::col2im(const Eigen::MatrixXd &col_matrix,
															 int batch_size, int input_height,
															 int input_width, int input_channels,
															 int filter_height, int filter_width,
															 int pad_h, int pad_w, 
															 int stride_h, int stride_w)
{
	Eigen::Tensor<double, 4> output_tensor(batch_size, input_height,
										   input_width, input_channels);
	output_tensor.setZero();

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
								output_tensor(b, h, w, c) += col_matrix(row_index, col_index); 
							}

							row_index++;
						}
					}
				}
				col_index++;
			}
		}
	}
	return output_tensor;
}