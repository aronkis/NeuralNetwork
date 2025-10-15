#include <random>
#include <cmath>
#include <cstring>
#include <iostream>

#include "Convolution.h"
#include "TensorUtils.h"

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

	biases_ = Eigen::Tensor<double, 1>(number_of_filters);
	biases_.setZero();

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

void NEURAL_NETWORK::Convolution::forward(const Eigen::Tensor<double, 2>& inputs, bool training)
{
	int batch_size = inputs.dimension(0);
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	// Convert input tensor to matrix, then to 4D tensor for convolution
	Eigen::MatrixXd inputs_matrix(inputs.dimension(0), inputs.dimension(1));
	for (int i = 0; i < inputs.dimension(0); i++)
	{
		for (int j = 0; j < inputs.dimension(1); j++)
		{
			inputs_matrix(i, j) = inputs(i, j);
		}
	}
	TensorUtils::MatrixToTensor4D(inputs_matrix, inputs_, batch_size, input_height_, input_width_, input_channels);

	im2col_input_ = im2col(inputs_,
						   filter_height, filter_width,
						   pad_height_, pad_width_,
						   stride_height_, stride_width_);

	const Eigen::Tensor<double, 2> filter_tensor = WeightsToTensor2D();

	// Perform tensor matrix multiplication: result = filter_tensor * im2col_input_
	Eigen::Tensor<double, 2> result(filter_tensor.dimension(0), im2col_input_.dimension(1));
	for (int i = 0; i < filter_tensor.dimension(0); i++)
	{
		for (int j = 0; j < im2col_input_.dimension(1); j++)
		{
			double sum = 0.0;
			for (int k = 0; k < filter_tensor.dimension(1); k++)
			{
				sum += filter_tensor(i, k) * im2col_input_(k, j);
			}
			result(i, j) = sum;
		}
	}

	// Add biases - broadcast across columns
	for (int i = 0; i < result.dimension(0); i++)
	{
		for (int j = 0; j < result.dimension(1); j++)
		{
			result(i, j) += biases_(i);
		}
	}

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

	// Convert convolution result to output tensor using row-major ordering
	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < output_height; h++) {
			for (int w = 0; w < output_width; w++) {
				int spatial_idx = b * output_height * output_width + h * output_width + w;
				for (int f = 0; f < num_filters; f++) {
					tensor_output_(b, h, w, f) = result(f, spatial_idx);
				}
			}
		}
	}

	Eigen::MatrixXd output_matrix = TensorUtils::Tensor4DToMatrix(tensor_output_);
	output_ = Eigen::Tensor<double, 2>(output_matrix.rows(), output_matrix.cols());
	for (int i = 0; i < output_matrix.rows(); i++)
	{
		for (int j = 0; j < output_matrix.cols(); j++)
		{
			output_(i, j) = output_matrix(i, j);
		}
	}
}

void NEURAL_NETWORK::Convolution::backward(const Eigen::Tensor<double, 2> &d_values)
{
	int batch_size = d_values.dimension(0);
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	int output_height = ((input_height_ + (2 * pad_height_) - filter_height) /
						 stride_height_) + 1;
	int output_width = ((input_width_ + (2 * pad_width_) - filter_width) /
						stride_width_) + 1;

	// Convert tensor to matrix for processing
	Eigen::MatrixXd d_values_matrix(d_values.dimension(0), d_values.dimension(1));
	for (int i = 0; i < d_values.dimension(0); i++)
	{
		for (int j = 0; j < d_values.dimension(1); j++)
		{
			d_values_matrix(i, j) = d_values(i, j);
		}
	}

	int total_spatial = batch_size * output_height * output_width;
	Eigen::Map<const Eigen::MatrixXd> d_values_reshaped(d_values_matrix.data(),
														num_filters, total_spatial);

	// Compute bias gradients - create tensor from matrix sum
	Eigen::VectorXd d_biases_vector = d_values_reshaped.rowwise().sum();
	d_biases_ = Eigen::Tensor<double, 1>(num_filters);
	for (int i = 0; i < num_filters; i++)
	{
		d_biases_(i) = d_biases_vector(i);
	}

	// Convert im2col_input_ to matrix for computation
	Eigen::MatrixXd im2col_matrix(im2col_input_.dimension(0), im2col_input_.dimension(1));
	for (int i = 0; i < im2col_input_.dimension(0); i++)
	{
		for (int j = 0; j < im2col_input_.dimension(1); j++)
		{
			im2col_matrix(i, j) = im2col_input_(i, j);
		}
	}
	Eigen::MatrixXd d_weights_matrix = im2col_matrix * d_values_reshaped.transpose();

	// Convert d_weights_matrix to tensor
	Eigen::Tensor<double, 2> d_weights_tensor(d_weights_matrix.rows(), d_weights_matrix.cols());
	for (int i = 0; i < d_weights_matrix.rows(); i++)
	{
		for (int j = 0; j < d_weights_matrix.cols(); j++)
		{
			d_weights_tensor(i, j) = d_weights_matrix(i, j);
		}
	}
	WeightsToTensor(d_weights_tensor);

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
		for (int i = 0; i < num_filters; i++)
		{
			d_biases_(i) += weight_regularizer_l1_ * (biases_(i) > 0 ? 1.0 : -1.0);
		}
	}

	if (bias_regularizer_l2_ > 0)
	{
		for (int i = 0; i < num_filters; i++)
		{
			d_biases_(i) += 2 * bias_regularizer_l2_ * biases_(i);
		}
	}

	Eigen::Tensor<double, 2> weights_tensor = WeightsToTensor2D();

	// Convert weights tensor to matrix for computation
	Eigen::MatrixXd weights_matrix(weights_tensor.dimension(0), weights_tensor.dimension(1));
	for (int i = 0; i < weights_tensor.dimension(0); i++)
	{
		for (int j = 0; j < weights_tensor.dimension(1); j++)
		{
			weights_matrix(i, j) = weights_tensor(i, j);
		}
	}

	Eigen::MatrixXd d_input_col = weights_matrix.transpose() * d_values_reshaped;

	// Convert d_input_col to tensor for col2im
	Eigen::Tensor<double, 2> d_input_col_tensor(d_input_col.rows(), d_input_col.cols());
	for (int i = 0; i < d_input_col.rows(); i++)
	{
		for (int j = 0; j < d_input_col.cols(); j++)
		{
			d_input_col_tensor(i, j) = d_input_col(i, j);
		}
	}

	col2im(d_input_col_tensor, d_values.dimension(0), input_height_,
		   input_width_, input_channels, filter_height,
		   filter_width, pad_height_, pad_width_,
		   stride_height_, stride_width_);

	Eigen::MatrixXd d_input_matrix = TensorUtils::Tensor4DToMatrix(d_input_tensor_);
	d_input_ = Eigen::Tensor<double, 2>(d_input_matrix.rows(), d_input_matrix.cols());
	for (int i = 0; i < d_input_matrix.rows(); i++)
	{
		for (int j = 0; j < d_input_matrix.cols(); j++)
		{
			d_input_(i, j) = d_input_matrix(i, j);
		}
	}
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Convolution::WeightsToTensor2D() const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	Eigen::Tensor<double, 2> weights_tensor(num_filters, filter_height * filter_width * input_channels);

	for (int f = 0; f < num_filters; f++)
	{
		int col = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					weights_tensor(f, col++) = weights_(h, w, c, f);
				}
			}
		}
	}

	return weights_tensor;
}

void NEURAL_NETWORK::Convolution::WeightsToTensor(const Eigen::Tensor<double, 2>& weights_tensor2d)
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

	if (weights_tensor2d.dimension(0) * weights_tensor2d.dimension(1) == expected_size &&
		weights_tensor2d.dimension(0) == filter_height * filter_width * input_channels &&
		weights_tensor2d.dimension(1) == num_filters)
	{
		Eigen::Map<Eigen::MatrixXd> weights_as_matrix(d_weights_.data(),
													  filter_height * filter_width * input_channels,
													  num_filters);
		// Copy tensor data to d_weights_ tensor directly
		for (int i = 0; i < weights_tensor2d.dimension(0); i++)
		{
			for (int j = 0; j < weights_tensor2d.dimension(1); j++)
			{
				weights_as_matrix(i, j) = weights_tensor2d(i, j);
			}
		}
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
						d_weights_(h, w, c, f) = weights_tensor2d(row++, f);
					}
				}
			}
		}
	}

}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Convolution::im2col(const Eigen::Tensor<double, 4> &input_tensor,
													int filter_height, int filter_width,
													int pad_h, int pad_w, int stride_h, int stride_w)
{
	// Get TensorUtils result and convert to tensor
	Eigen::MatrixXd matrix_result = TensorUtils::im2col(input_tensor, filter_height, filter_width,
								pad_h, pad_w, stride_h, stride_w);

	// Convert MatrixXd to Tensor<double, 2>
	Eigen::Tensor<double, 2> tensor_result(matrix_result.rows(), matrix_result.cols());
	for (int i = 0; i < matrix_result.rows(); i++)
	{
		for (int j = 0; j < matrix_result.cols(); j++)
		{
			tensor_result(i, j) = matrix_result(i, j);
		}
	}
	return tensor_result;
}

void NEURAL_NETWORK::Convolution::col2im(const Eigen::Tensor<double, 2> &col_tensor,
										  int batch_size, int input_height,
										  int input_width, int input_channels,
										  int filter_height, int filter_width,
										  int pad_h, int pad_w,
										  int stride_h, int stride_w)
{
	// Convert tensor to matrix for TensorUtils call
	Eigen::MatrixXd col_matrix(col_tensor.dimension(0), col_tensor.dimension(1));
	for (int i = 0; i < col_tensor.dimension(0); i++)
	{
		for (int j = 0; j < col_tensor.dimension(1); j++)
		{
			col_matrix(i, j) = col_tensor(i, j);
		}
	}

	TensorUtils::col2im(col_matrix, d_input_tensor_, batch_size, input_height,
						input_width, input_channels, filter_height, filter_width,
						pad_h, pad_w, stride_h, stride_w);
}

// Helper methods for tensor conversions
Eigen::Tensor<double, 2> NEURAL_NETWORK::Convolution::WeightsTensor2DFromTensor4D(const Eigen::Tensor<double, 4>& tensor) const
{
	int filter_height = tensor.dimension(0);
	int filter_width = tensor.dimension(1);
	int input_channels = tensor.dimension(2);
	int num_filters = tensor.dimension(3);

	Eigen::Tensor<double, 2> tensor2d(num_filters, filter_height * filter_width * input_channels);

	for (int f = 0; f < num_filters; f++)
	{
		int col = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					tensor2d(f, col++) = tensor(h, w, c, f);
				}
			}
		}
	}

	return tensor2d;
}

void NEURAL_NETWORK::Convolution::Tensor2DToWeightsTensor4D(const Eigen::Tensor<double, 2>& tensor2d, Eigen::Tensor<double, 4>& tensor4d) const
{
	int filter_height = weights_.dimension(0);
	int filter_width = weights_.dimension(1);
	int input_channels = weights_.dimension(2);
	int num_filters = weights_.dimension(3);

	if (tensor4d.size() == 0)
	{
		tensor4d = Eigen::Tensor<double, 4>(filter_height, filter_width, input_channels, num_filters);
	}

	for (int f = 0; f < num_filters; f++)
	{
		int col = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					tensor4d(h, w, c, f) = tensor2d(f, col++);
				}
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

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetWeights() const
{
	weights_matrix_cache_ = WeightsToTensor2D();
	return weights_matrix_cache_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::Convolution::GetBiases() const
{
	return biases_;
}

std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>> NEURAL_NETWORK::Convolution::GetParameters() const
{
	return std::make_pair(WeightsToTensor2D(), biases_);
}

void NEURAL_NETWORK::Convolution::SetParameters(const Eigen::Tensor<double, 2>& weights,
												const Eigen::Tensor<double, 1>& biases)
{
	// Convert 2D weights tensor to 4D weights tensor
	Tensor2DToWeightsTensor4D(weights, weights_);

	// Copy biases directly
	biases_ = biases;
}

//maybe weights in the n-th filter? GetWeight(int n)
const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Convolution::GetWeightsTensor() const
{
	return weights_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::Convolution::GetBiasesVector() const
{
	return biases_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetOutput() const
{
	return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetDInput() const
{
	return d_input_;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Convolution::predictions() const
{
	return output_;
}

void NEURAL_NETWORK::Convolution::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
	d_input_ = dinput;
}

// LayerBase virtual method implementations for gradients
const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetDWeights() const
{
	d_weights_matrix_cache_ = WeightsTensor2DFromTensor4D(d_weights_);
	return d_weights_matrix_cache_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::Convolution::GetDBiases() const
{
	return d_biases_;
}

// LayerBase virtual method implementations for momentums/caches (Tensor interface)
const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetWeightMomentums() const
{
	weight_momentums_matrix_cache_ = WeightsTensor2DFromTensor4D(weight_momentums_);
	return weight_momentums_matrix_cache_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::Convolution::GetBiasMomentums() const
{
	return bias_momentums_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Convolution::GetWeightCaches() const
{
	weight_caches_matrix_cache_ = WeightsTensor2DFromTensor4D(weight_caches_);
	return weight_caches_matrix_cache_;
}

const Eigen::Tensor<double, 1>& NEURAL_NETWORK::Convolution::GetBiasCaches() const
{
	return bias_caches_;
}

void NEURAL_NETWORK::Convolution::SetWeightMomentums(const Eigen::Tensor<double, 2>& weight_momentums)
{
	Tensor2DToWeightsTensor4D(weight_momentums, weight_momentums_);
}

void NEURAL_NETWORK::Convolution::SetBiasMomentums(const Eigen::Tensor<double, 1>& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::Convolution::SetWeightCaches(const Eigen::Tensor<double, 2>& weight_caches)
{
	Tensor2DToWeightsTensor4D(weight_caches, weight_caches_);
}

void NEURAL_NETWORK::Convolution::SetBiasCaches(const Eigen::Tensor<double, 1>& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::Convolution::UpdateWeights(Eigen::Tensor<double, 2>& weight_update)
{
	Eigen::Tensor<double, 4> weight_update_tensor;
	Tensor2DToWeightsTensor4D(weight_update, weight_update_tensor);
	UpdateWeightsTensor(weight_update_tensor);
}

void NEURAL_NETWORK::Convolution::UpdateWeightsCache(Eigen::Tensor<double, 2>& weight_update)
{
	Eigen::Tensor<double, 4> weight_update_tensor;
	Tensor2DToWeightsTensor4D(weight_update, weight_update_tensor);
	UpdateWeightsCacheTensor(weight_update_tensor);
}

void NEURAL_NETWORK::Convolution::UpdateBiases(Eigen::Tensor<double, 1>& bias_update)
{
	UpdateBiasesTensor(bias_update);
}

void NEURAL_NETWORK::Convolution::UpdateBiasesCache(Eigen::Tensor<double, 1>& bias_update)
{
	UpdateBiasesCacheTensor(bias_update);
}

void NEURAL_NETWORK::Convolution::SetWeightMomentumsTensor(const Eigen::Tensor<double, 4>& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::Convolution::SetBiasMomentumsTensor(const Eigen::Tensor<double, 1>& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::Convolution::SetWeightCachesTensor(const Eigen::Tensor<double, 4>& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::Convolution::SetBiasCachesTensor(const Eigen::Tensor<double, 1>& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::Convolution::UpdateWeightsTensor(Eigen::Tensor<double, 4>& weight_update)
{
	weights_ = weights_ + weight_update;
}

void NEURAL_NETWORK::Convolution::UpdateWeightsCacheTensor(Eigen::Tensor<double, 4>& weight_update)
{
	weight_caches_ = weight_caches_ + weight_update;
}

void NEURAL_NETWORK::Convolution::UpdateBiasesTensor(Eigen::Tensor<double, 1>& bias_update)
{
	biases_ = biases_ + bias_update;
}

void NEURAL_NETWORK::Convolution::UpdateBiasesCacheTensor(Eigen::Tensor<double, 1>& bias_update)
{
	bias_caches_ = bias_caches_ + bias_update;
}
