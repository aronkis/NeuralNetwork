#include <random>
#include <cmath>
#include <cstring>

#include "Convolution.h"

NEURAL_NETWORK::Convolution::Convolution(int number_of_filters, int filter_height, int filter_width,
										 int input_height, int input_width, int input_channels,
										 int padding, int stride_height, int stride_width,
										 double weight_regularizer_l1, double weight_regularizer_l2,
										 double bias_regularizer_l1, double bias_regularizer_l2)
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
	input_channels_ = input_channels;
	filter_height_ = filter_height;
	filter_width_ = filter_width;
	number_of_filters_ = number_of_filters;
	padding_ = padding;
	stride_height_ = stride_height;
	stride_width_ = stride_width;

	pad_height_ = padding_ ? (filter_height - 1) / 2 : 0;
	pad_width_ = padding_ ? (filter_width - 1) / 2 : 0;

	weight_regularizer_l1_ = weight_regularizer_l1;
	weight_regularizer_l2_ = weight_regularizer_l2;
	bias_regularizer_l1_ = bias_regularizer_l1;
	bias_regularizer_l2_ = bias_regularizer_l2;
}

void NEURAL_NETWORK::Convolution::forward(const Eigen::MatrixXd& inputs, bool training)
{
	int batch_size = inputs.rows();
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	InputMatrixToTensor(inputs, batch_size, input_height_, input_width_, input_channels);

	im2col_input_ = im2col(inputs_,
						   filter_height, filter_width,
						   pad_height_, pad_width_,
						   stride_height_, stride_width_);

	const Eigen::MatrixXd filter_matrix = WeightsToMatrix();

	Eigen::MatrixXd result = filter_matrix * im2col_input_; // Convolution

	result.colwise() += biases_; 

	int output_height = ((input_height_ + (2 * pad_height_) - filter_height) /
						  stride_height_) + 1;
	int output_width =  ((input_width_ + (2 * pad_width_) - filter_width) /
						  stride_width_) + 1;

	if (tensor_output_.size() == 0 ||
		tensor_output_.dimension(0) != batch_size ||
		tensor_output_.dimension(1) != output_height ||
		tensor_output_.dimension(2) != output_width ||
		tensor_output_.dimension(3) != num_filters) 
	{
		tensor_output_ = Eigen::Tensor<double, 4>(batch_size, output_height, output_width, num_filters);
	}

	int total_spatial = output_height * output_width * batch_size;
	Eigen::Map<Eigen::MatrixXd> tensor_map(tensor_output_.data(), num_filters, total_spatial); // Transposed tensor map

	tensor_map = result;

	output_ = InputTensorToMatrix(tensor_output_);
}

void NEURAL_NETWORK::Convolution::backward(const Eigen::MatrixXd &d_values)
{
	int batch_size = d_values.rows();
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	int output_height = ((input_height_ + (2 * pad_height_) - filter_height) /
						 stride_height_) + 1;
	int output_width = ((input_width_ + (2 * pad_width_) - filter_width) /
						stride_width_) + 1;

	int total_spatial = batch_size * output_height * output_width;
	Eigen::Map<const Eigen::MatrixXd> d_values_reshaped(d_values.data(),
														num_filters, total_spatial);

	d_biases_ = d_values_reshaped.rowwise().sum();

	Eigen::MatrixXd d_weights_matrix = im2col_input_ * d_values_reshaped.transpose();
	
	WeightsToTensor(d_weights_matrix);

	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::MatrixXd d_input_col = weights_matrix.transpose() * d_values_reshaped;

	col2im(d_input_col, d_values.rows(), input_height_,
		   input_width_, input_channels, filter_height,
		   filter_width, pad_height_, pad_width_,
		   stride_height_, stride_width_);
	d_input_ = InputTensorToMatrix(d_input_tensor_);
}

//maybe weights in the n-th filter? GetWeight(int n)
const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Convolution::GetWeightsTensor() const
{
	return weights_;
}

const Eigen::VectorXd& NEURAL_NETWORK::Convolution::GetBiasesVector() const
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

// ADAM optimizer support functions
void NEURAL_NETWORK::Convolution::SetWeightMomentums(const Eigen::Tensor<double, 4>& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::Convolution::SetBiasMomentums(const Eigen::VectorXd& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::Convolution::SetWeightCaches(const Eigen::Tensor<double, 4>& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::Convolution::SetBiasCaches(const Eigen::VectorXd& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::Convolution::UpdateWeights(Eigen::Tensor<double, 4>& weight_update)
{
	weight_momentums_ = weight_update;
}

void NEURAL_NETWORK::Convolution::UpdateWeightsCache(Eigen::Tensor<double, 4>& weight_update)
{
	weight_caches_ = weight_update;
}

void NEURAL_NETWORK::Convolution::UpdateBiases(Eigen::VectorXd& bias_update)
{
	bias_momentums_ = bias_update;
}

void NEURAL_NETWORK::Convolution::UpdateBiasesCache(Eigen::VectorXd& bias_update)
{
	bias_caches_ = bias_update;
}

void NEURAL_NETWORK::Convolution::InputMatrixToTensor(const Eigen::MatrixXd& matrix,
												  int batch_size, int height, int width, int channels)
{
	if (inputs_.size() == 0 || inputs_.dimension(0) != batch_size) {
		inputs_ = Eigen::Tensor<double, 4>(batch_size, height, width, channels);
	}

	auto expected_size = batch_size * height * width * channels;

	if (matrix.rows() == batch_size &&
		matrix.cols() == height * width * channels &&
		matrix.size() == expected_size)
	{
		Eigen::Map<Eigen::MatrixXd> tensor_as_matrix(inputs_.data(), batch_size, height * width * channels);
		tensor_as_matrix = matrix;
	}
	else
	{
		for (int b = 0; b < batch_size && b < matrix.rows(); b++)
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
							inputs_(b, h, w, c) = matrix(b, matrix_idx);
						}
					}
				}
			}
		}
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::InputTensorToMatrix(const Eigen::Tensor<double, 4>& tensor)
{
	int batch_size = tensor.dimension(0);
	int height = tensor.dimension(1);
	int width = tensor.dimension(2);
	int channels = tensor.dimension(3);
	auto expected_size = batch_size * height * width * channels;

	if (tensor.size() == expected_size)
	{
		return Eigen::Map<const Eigen::MatrixXd>(tensor.data(),
												 batch_size,
												 height * width * channels);
	}
	else
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
						int matrix_idx = h * width * channels + w * channels + c;
						matrix(b, matrix_idx) = tensor(b, h, w, c);
					}
				}
			}
		}
		return matrix;
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution::WeightsToMatrix() const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	auto expected_size = filter_height * filter_width * input_channels * num_filters;

	if (weights_.size() == expected_size)
	{
		Eigen::Map<const Eigen::MatrixXd> weights_map(weights_.data(),
													  num_filters,
													  filter_height * filter_width * input_channels); // Transposed tensor map
		return weights_map;
	}
	else
	{
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
}

void NEURAL_NETWORK::Convolution::WeightsToTensor(const Eigen::MatrixXd& weights_matrix)
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	if (d_weights_.size() == 0) 
	{
		d_weights_ = Eigen::Tensor<double, 4>(filter_height, filter_width, input_channels, num_filters);
	}

	auto expected_size = filter_height * filter_width * input_channels * num_filters;

	if (weights_matrix.rows() * weights_matrix.cols() == expected_size &&
		weights_matrix.rows() == filter_height * filter_width * input_channels &&
		weights_matrix.cols() == num_filters)
	{
		Eigen::Map<Eigen::MatrixXd> weights_as_matrix(d_weights_.data(),
													  filter_height * filter_width * input_channels,
													  num_filters);
		weights_as_matrix = weights_matrix;
	}
	else
	{
		for (int f = 0; f < num_filters; f++)
		{
			int row = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int h = 0; h < filter_height; h++)
				{
					for (int w = 0; w < filter_width; w++)
					{
						d_weights_(h, w, c, f) = weights_matrix(row++, f);
					}
				}
			}
		}
	}

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

void NEURAL_NETWORK::Convolution::col2im(const Eigen::MatrixXd &col_matrix,
										  int batch_size, int input_height,
										  int input_width, int input_channels,
										  int filter_height, int filter_width,
										  int pad_h, int pad_w,
										  int stride_h, int stride_w)
{
	if (d_input_tensor_.size() == 0 ||
		d_input_tensor_.dimension(0) != batch_size ||
		d_input_tensor_.dimension(1) != input_height ||
		d_input_tensor_.dimension(2) != input_width ||
		d_input_tensor_.dimension(3) != input_channels)
	{
		d_input_tensor_ = Eigen::Tensor<double, 4>(batch_size, input_height, input_width, input_channels);
	}

	d_input_tensor_.setZero();

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
								d_input_tensor_(b, h, w, c) += col_matrix(row_index, col_index);
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

// Getter methods for convolution parameters
int NEURAL_NETWORK::Convolution::GetNumberOfFilters() const
{
	return number_of_filters_;
}

int NEURAL_NETWORK::Convolution::GetFilterHeight() const
{
	return filter_height_;
}

int NEURAL_NETWORK::Convolution::GetFilterWidth() const
{
	return filter_width_;
}

int NEURAL_NETWORK::Convolution::GetInputHeight() const
{
	return input_height_;
}

int NEURAL_NETWORK::Convolution::GetInputWidth() const
{
	return input_width_;
}

int NEURAL_NETWORK::Convolution::GetInputChannels() const
{
	return input_channels_;
}

int NEURAL_NETWORK::Convolution::GetPadding() const
{
	return padding_;
}

int NEURAL_NETWORK::Convolution::GetStrideHeight() const
{
	return stride_height_;
}

int NEURAL_NETWORK::Convolution::GetStrideWidth() const
{
	return stride_width_;
}

double NEURAL_NETWORK::Convolution::GetWeightRegularizerL1() const
{
	return weight_regularizer_l1_;
}

double NEURAL_NETWORK::Convolution::GetWeightRegularizerL2() const
{
	return weight_regularizer_l2_;
}

double NEURAL_NETWORK::Convolution::GetBiasRegularizerL1() const
{
	return bias_regularizer_l1_;
}

double NEURAL_NETWORK::Convolution::GetBiasRegularizerL2() const
{
	return bias_regularizer_l2_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution::GetWeights() const
{
	weights_matrix_cache_ = WeightsToMatrix();
	return weights_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution::GetBiases() const
{
	biases_cache_ = biases_.transpose();
	return biases_cache_;
}

std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> NEURAL_NETWORK::Convolution::GetParameters() const
{
	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::RowVectorXd biases_row = biases_.transpose();

	return std::make_pair(weights_matrix, biases_row);
}

void NEURAL_NETWORK::Convolution::SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases)
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	auto expected_size = filter_height * filter_width * input_channels * num_filters;

	if (weights.size() == expected_size &&
		weights.rows() == num_filters &&
		weights.cols() == filter_height * filter_width * input_channels)
	{
		std::memcpy(weights_.data(), weights.data(), expected_size * sizeof(double));
	}
	else
	{
		for (int f = 0; f < num_filters; f++)
		{
			int col = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int h = 0; h < filter_height; h++)
				{
					for (int w = 0; w < filter_width; w++)
					{
						weights_(h, w, c, f) = weights(f, col++);
					}
				}
			}
		}
	}

	biases_ = biases.transpose();
}