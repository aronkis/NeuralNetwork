#include <random>
#include <cmath>
#include <cstring>
#include <iostream>

#include "Convolution2D.h"
#include "TensorUtils.h"

NEURAL_NETWORK::Convolution2D::Convolution2D(int number_of_filters, int filter_height, int filter_width,
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

void NEURAL_NETWORK::Convolution2D::forward(const Eigen::MatrixXd& inputs, bool training)
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

	Eigen::MatrixXd result = filter_matrix * im2col_input_; // Convolution2D

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

	// Convert Convolution2D result to output tensor using row-major ordering
	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < output_height; h++) 
		{
			for (int w = 0; w < output_width; w++) 
			{
				int spatial_idx = b * output_height * output_width + h * output_width + w;
				for (int f = 0; f < num_filters; f++) 
				 
				{
					tensor_output_(b, h, w, f) = result(f, spatial_idx);
				}
			}
		}
	}

	output_ = InputTensorToMatrix(tensor_output_);
}

void NEURAL_NETWORK::Convolution2D::backward(const Eigen::MatrixXd &d_values)
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

	// Add regularization to d_weights_ tensor
	if (weight_regularizer_l1_ > 0)
	{
		d_weights_ = d_weights_ + weight_regularizer_l1_ * weights_.sign();
	}

	if (weight_regularizer_l2_ > 0)
	{
		d_weights_ = d_weights_ + 2 * weight_regularizer_l2_ * weights_;
	}

	// Add regularization to d_biases_
	if (bias_regularizer_l1_ > 0)
	{
		d_biases_.array() += bias_regularizer_l1_ * biases_.array().sign();
	}

	if (bias_regularizer_l2_ > 0)
	{
		d_biases_.array() += 2 * bias_regularizer_l2_ * biases_.array();
	}

	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::MatrixXd d_input_col = weights_matrix.transpose() * d_values_reshaped;

	col2im(d_input_col, d_values.rows(), input_height_,
		   input_width_, input_channels, filter_height,
		   filter_width, pad_height_, pad_width_,
		   stride_height_, stride_width_);
	d_input_ = InputTensorToMatrix(d_input_tensor_);
}

void NEURAL_NETWORK::Convolution2D::InputMatrixToTensor(const Eigen::MatrixXd& matrix,
												  int batch_size, int height, int width, int channels)
{
	TensorUtils::MatrixToTensor4D(matrix, inputs_, batch_size, height, width, channels);
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution2D::InputTensorToMatrix(const Eigen::Tensor<double, 4>& tensor)
{
	return TensorUtils::Tensor4DToMatrix(tensor);
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution2D::WeightsToMatrix() const
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

void NEURAL_NETWORK::Convolution2D::WeightsToTensor(const Eigen::MatrixXd& weights_matrix)
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

Eigen::MatrixXd NEURAL_NETWORK::Convolution2D::im2col(const Eigen::Tensor<double, 4> &input_tensor,
													int filter_height, int filter_width,
													int pad_h, int pad_w, int stride_h, int stride_w)
{
	return TensorUtils::im2col(input_tensor, filter_height, filter_width,
								pad_h, pad_w, stride_h, stride_w);
}

void NEURAL_NETWORK::Convolution2D::col2im(const Eigen::MatrixXd &col_matrix,
										  int batch_size, int input_height,
										  int input_width, int input_channels,
										  int filter_height, int filter_width,
										  int pad_h, int pad_w,
										  int stride_h, int stride_w)
{
	TensorUtils::col2im(col_matrix, d_input_tensor_, batch_size, input_height, 
						input_width, input_channels, filter_height, filter_width, 
						pad_h, pad_w, stride_h, stride_w);
}

// Helper methods for tensor-matrix conversions
Eigen::MatrixXd NEURAL_NETWORK::Convolution2D::WeightsToMatrixFromTensor(const Eigen::Tensor<double, 4>& tensor) const
{
	int filter_height = tensor.dimension(0);
	int filter_width = tensor.dimension(1);
	int input_channels = tensor.dimension(2);
	int num_filters = tensor.dimension(3);

	auto expected_size = filter_height * filter_width * input_channels * num_filters;

	if (tensor.size() == expected_size)
	{
		Eigen::Map<const Eigen::MatrixXd> tensor_map(tensor.data(),
													 num_filters,
													 filter_height * filter_width * input_channels);
		return tensor_map;
	}
	else
	{
		Eigen::MatrixXd matrix(num_filters, filter_height * filter_width * input_channels);

		for (int f = 0; f < num_filters; f++)
		{
			int col = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int h = 0; h < filter_height; h++)
				{
					for (int w = 0; w < filter_width; w++)
					{
						matrix(f, col++) = tensor(h, w, c, f);
					}
				}
			}
		}

		return matrix;
	}
}

void NEURAL_NETWORK::Convolution2D::MatrixToWeightsTensor(const Eigen::MatrixXd& matrix, 
														Eigen::Tensor<double, 4>& tensor) const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	if (tensor.size() == 0) 
	{
		tensor = Eigen::Tensor<double, 4>(filter_height, filter_width, input_channels, num_filters);
	}

	auto expected_size = filter_height * filter_width * input_channels * num_filters;

	if (matrix.rows() * matrix.cols() == expected_size &&
		matrix.rows() == num_filters &&
		matrix.cols() == filter_height * filter_width * input_channels)
	{
		Eigen::Map<Eigen::MatrixXd> tensor_as_matrix(tensor.data(),
													 num_filters,
													 filter_height * filter_width * input_channels);
		tensor_as_matrix = matrix;
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
						tensor(h, w, c, f) = matrix(f, col++);
					}
				}
			}
		}
	}
}

int NEURAL_NETWORK::Convolution2D::GetNumberOfFilters() const
{
	return number_of_filters_;
}

int NEURAL_NETWORK::Convolution2D::GetFilterHeight() const
{
	return filter_height_;
}

int NEURAL_NETWORK::Convolution2D::GetFilterWidth() const
{
	return filter_width_;
}

int NEURAL_NETWORK::Convolution2D::GetInputHeight() const
{
	return input_height_;
}

int NEURAL_NETWORK::Convolution2D::GetInputWidth() const
{
	return input_width_;
}

int NEURAL_NETWORK::Convolution2D::GetInputChannels() const
{
	return input_channels_;
}

int NEURAL_NETWORK::Convolution2D::GetPadding() const
{
	return padding_;
}

int NEURAL_NETWORK::Convolution2D::GetStrideHeight() const
{
	return stride_height_;
}

int NEURAL_NETWORK::Convolution2D::GetStrideWidth() const
{
	return stride_width_;
}

double NEURAL_NETWORK::Convolution2D::GetWeightRegularizerL1() const
{
	return weight_regularizer_l1_;
}

double NEURAL_NETWORK::Convolution2D::GetWeightRegularizerL2() const
{
	return weight_regularizer_l2_;
}

double NEURAL_NETWORK::Convolution2D::GetBiasRegularizerL1() const
{
	return bias_regularizer_l1_;
}

double NEURAL_NETWORK::Convolution2D::GetBiasRegularizerL2() const
{
	return bias_regularizer_l2_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetWeights() const
{
	weights_matrix_cache_ = WeightsToMatrix();
	return weights_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution2D::GetBiases() const
{
	biases_cache_ = biases_.transpose();
	return biases_cache_;
}

std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> NEURAL_NETWORK::Convolution2D::GetParameters() const
{
	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::RowVectorXd biases_row = biases_.transpose();

	return std::make_pair(weights_matrix, biases_row);
}

void NEURAL_NETWORK::Convolution2D::SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases)
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

//maybe weights in the n-th filter? GetWeight(int n)
const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Convolution2D::GetWeightsTensor() const
{
	return weights_;
}

const Eigen::VectorXd& NEURAL_NETWORK::Convolution2D::GetBiasesVector() const
{
	return biases_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetDInput() const
{
	return d_input_;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution2D::predictions() const
{
	return output_;
}

void NEURAL_NETWORK::Convolution2D::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_input_ = dinput;
}

// LayerBase virtual method implementations for gradients
const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetDWeights() const
{
	d_weights_matrix_cache_ = WeightsToMatrixFromTensor(d_weights_);
	return d_weights_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution2D::GetDBiases() const
{
	d_biases_cache_ = d_biases_.transpose();
	return d_biases_cache_;
}

// LayerBase virtual method implementations for momentums/caches (Matrix/RowVector interface)
const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetWeightMomentums() const
{
	weight_momentums_matrix_cache_ = WeightsToMatrixFromTensor(weight_momentums_);
	return weight_momentums_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution2D::GetBiasMomentums() const
{
	bias_momentums_cache_ = bias_momentums_.transpose();
	return bias_momentums_cache_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution2D::GetWeightCaches() const
{
	weight_caches_matrix_cache_ = WeightsToMatrixFromTensor(weight_caches_);
	return weight_caches_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution2D::GetBiasCaches() const
{
	bias_caches_cache_ = bias_caches_.transpose();
	return bias_caches_cache_;
}

void NEURAL_NETWORK::Convolution2D::SetWeightMomentums(const Eigen::MatrixXd& weight_momentums)
{
	Eigen::Tensor<double, 4> weight_momentums_tensor;
	MatrixToWeightsTensor(weight_momentums, weight_momentums_tensor);
	SetWeightMomentumsTensor(weight_momentums_tensor);
}

void NEURAL_NETWORK::Convolution2D::SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums)
{
	Eigen::VectorXd bias_momentums_vector = bias_momentums.transpose();
	SetBiasMomentumsTensor(bias_momentums_vector);
}

void NEURAL_NETWORK::Convolution2D::SetWeightCaches(const Eigen::MatrixXd& weight_caches)
{
	Eigen::Tensor<double, 4> weight_caches_tensor;
	MatrixToWeightsTensor(weight_caches, weight_caches_tensor);
	SetWeightCachesTensor(weight_caches_tensor);
}

void NEURAL_NETWORK::Convolution2D::SetBiasCaches(const Eigen::RowVectorXd& bias_caches)
{
	Eigen::VectorXd bias_caches_vector = bias_caches.transpose();
	SetBiasCachesTensor(bias_caches_vector);
}

void NEURAL_NETWORK::Convolution2D::UpdateWeights(Eigen::MatrixXd& weight_update)
{
	Eigen::Tensor<double, 4> weight_update_tensor;
	MatrixToWeightsTensor(weight_update, weight_update_tensor);
	UpdateWeightsTensor(weight_update_tensor);
}

void NEURAL_NETWORK::Convolution2D::UpdateWeightsCache(Eigen::MatrixXd& weight_update)
{
	Eigen::Tensor<double, 4> weight_update_tensor;
	MatrixToWeightsTensor(weight_update, weight_update_tensor);
	UpdateWeightsCacheTensor(weight_update_tensor);
}

void NEURAL_NETWORK::Convolution2D::UpdateBiases(Eigen::RowVectorXd& bias_update)
{
	Eigen::VectorXd bias_update_vector = bias_update.transpose();
	UpdateBiasesTensor(bias_update_vector);
}

void NEURAL_NETWORK::Convolution2D::UpdateBiasesCache(Eigen::RowVectorXd& bias_update)
{
	Eigen::VectorXd bias_update_vector = bias_update.transpose();
	UpdateBiasesCacheTensor(bias_update_vector);
}

void NEURAL_NETWORK::Convolution2D::SetWeightMomentumsTensor(const Eigen::Tensor<double, 4>& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::Convolution2D::SetBiasMomentumsTensor(const Eigen::VectorXd& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::Convolution2D::SetWeightCachesTensor(const Eigen::Tensor<double, 4>& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::Convolution2D::SetBiasCachesTensor(const Eigen::VectorXd& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::Convolution2D::UpdateWeightsTensor(Eigen::Tensor<double, 4>& weight_update)
{
	weights_ = weights_ + weight_update;
}

void NEURAL_NETWORK::Convolution2D::UpdateWeightsCacheTensor(Eigen::Tensor<double, 4>& weight_update)
{
	weight_caches_ = weight_caches_ + weight_update;
}

void NEURAL_NETWORK::Convolution2D::UpdateBiasesTensor(Eigen::VectorXd& bias_update)
{
	biases_ = biases_ + bias_update;
}

void NEURAL_NETWORK::Convolution2D::UpdateBiasesCacheTensor(Eigen::VectorXd& bias_update)
{
	bias_caches_ = bias_caches_ + bias_update;
}
