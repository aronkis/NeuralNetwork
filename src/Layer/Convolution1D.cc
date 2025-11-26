#include "Convolution1D.h"
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>

NEURAL_NETWORK::Convolution1D::Convolution1D(int number_of_filters, int filter_length,
											 int input_length, int input_channels,
											 int padding, int stride,
											 double weight_regularizer_l1, double weight_regularizer_l2,
											 double bias_regularizer_l1, double bias_regularizer_l2)
{
	std::mt19937 gen(0);
	// He normal initialization
	const int fan_in = filter_length * input_channels;
	const double he_std = std::sqrt(2.0 / static_cast<double>(fan_in));
	std::normal_distribution<> he_dist(0.0, he_std);

	weights_ = Eigen::Tensor<double, 3>(filter_length, input_channels, number_of_filters);
	weights_ = weights_.unaryExpr([&](double) { return he_dist(gen); });

	biases_ = Eigen::VectorXd::Zero(number_of_filters);

	input_length_ = input_length;
	input_channels_ = input_channels;
	filter_length_ = filter_length;
	number_of_filters_ = number_of_filters;
	padding_ = padding;
	stride_ = stride;

	pad_length_ = padding_ ? (filter_length - 1) / 2 : 0;

	weight_regularizer_l1_ = weight_regularizer_l1;
	weight_regularizer_l2_ = weight_regularizer_l2;
	bias_regularizer_l1_ = bias_regularizer_l1;
	bias_regularizer_l2_ = bias_regularizer_l2;
}


void NEURAL_NETWORK::Convolution1D::forward(const Eigen::MatrixXd& inputs, bool training)
{
	int batch_size = inputs.rows();
	int filter_length = weights_.dimension(0);
	int input_channels = weights_.dimension(1);
	int num_filters = weights_.dimension(2);

	InputMatrixToTensor(inputs, batch_size, input_length_, input_channels_);

	im2col_input_ = im2col(inputs_, filter_length, pad_length_, stride_);

	const Eigen::MatrixXd filter_matrix = WeightsToMatrix();
	Eigen::MatrixXd result = filter_matrix * im2col_input_;
	result.colwise() += biases_;

	int output_length = ((input_length_ + (2 * pad_length_) - filter_length) / stride_) + 1;

	if (tensor_output_.size() == 0 ||
		tensor_output_.dimension(0) != batch_size ||
		tensor_output_.dimension(1) != output_length ||
		tensor_output_.dimension(2) != num_filters)
	{
		tensor_output_ = Eigen::Tensor<double, 3>(batch_size, output_length, num_filters);
	}

	for (int b = 0; b < batch_size; b++)
	{
		for (int t = 0; t < output_length; t++)
		{
			int temporal_idx = b * output_length + t;
			for (int f = 0; f < num_filters; f++)
			{
				tensor_output_(b, t, f) = result(f, temporal_idx);
			}
		}
	}

	output_ = InputTensorToMatrix(tensor_output_);
}

void NEURAL_NETWORK::Convolution1D::backward(const Eigen::MatrixXd& dvalues)
{
	int batch_size = dvalues.rows();
	int filter_length = weights_.dimension(0);
	int input_channels = weights_.dimension(1);
	int num_filters = weights_.dimension(2);

	int output_length = ((input_length_ + (2 * pad_length_) - filter_length) / stride_) + 1;

	Eigen::MatrixXd d_result(num_filters, batch_size * output_length);

	for (int b = 0; b < batch_size; b++)
	{
		for (int t = 0; t < output_length; t++)
		{
			int temporal_idx = b * output_length + t;
			for (int f = 0; f < num_filters; f++)
			{
				int col_idx = t * num_filters + f;
				d_result(f, temporal_idx) = dvalues(b, col_idx);
			}
		}
	}

	d_biases_ = d_result.rowwise().sum();

	Eigen::MatrixXd d_weights_raw = im2col_input_ * d_result.transpose();

	Eigen::MatrixXd d_weights_matrix = d_weights_raw.transpose();

	WeightsToTensor(d_weights_matrix);

	if (weight_regularizer_l1_ > 0)
	{
		d_weights_ = d_weights_ + weight_regularizer_l1_ * weights_.sign();
	}

	if (weight_regularizer_l2_ > 0)
	{
		d_weights_ = d_weights_ + 2 * weight_regularizer_l2_ * weights_;
	}

	if (bias_regularizer_l1_ > 0)
	{
		d_biases_.array() += bias_regularizer_l1_ * biases_.array().sign();
	}

	if (bias_regularizer_l2_ > 0)
	{
		d_biases_.array() += 2 * bias_regularizer_l2_ * biases_.array();
	}

	Eigen::MatrixXd weights_matrix = WeightsToMatrix();
	Eigen::MatrixXd d_input_col = weights_matrix.transpose() * d_result;

	col2im(d_input_col, batch_size, input_length_, input_channels,
		   filter_length, pad_length_, stride_);

	d_input_ = InputTensorToMatrix(d_input_tensor_);
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetDInput() const
{
	return d_input_;
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution1D::predictions() const
{
	return output_;
}

double NEURAL_NETWORK::Convolution1D::GetWeightRegularizerL1() const
{ 
	return weight_regularizer_l1_; 
}

double NEURAL_NETWORK::Convolution1D::GetWeightRegularizerL2() const
{ 
	return weight_regularizer_l2_; 
}

double NEURAL_NETWORK::Convolution1D::GetBiasRegularizerL1() const
{ 
	return bias_regularizer_l1_; 
}

double NEURAL_NETWORK::Convolution1D::GetBiasRegularizerL2() const
{ 
	return bias_regularizer_l2_; 
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetWeights() const
{
	weights_matrix_cache_ = WeightsToMatrix();
	return weights_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution1D::GetBiases() const
{
	biases_cache_ = biases_.transpose();
	return biases_cache_;
}

std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> NEURAL_NETWORK::Convolution1D::GetParameters() const
{
	return { GetWeights(), GetBiases() };
}

void NEURAL_NETWORK::Convolution1D::SetParameters(const Eigen::MatrixXd& weights,
								  const Eigen::RowVectorXd& biases)
{
	int filter_length = weights_.dimension(0);
	int input_channels = weights_.dimension(1);
	int num_filters = weights_.dimension(2);

	auto expected_size = filter_length * input_channels * num_filters;

	if (weights.size() == expected_size &&
		weights.rows() == num_filters &&
		weights.cols() == filter_length * input_channels)
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
				for (int k = 0; k < filter_length; k++)
				{
					weights_(k, c, f) = weights(f, col++);
				}
			}
		}
	}

	biases_ = biases.transpose();
}

void NEURAL_NETWORK::Convolution1D::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_input_ = dinput;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetDWeights() const
{
	d_weights_matrix_cache_ = WeightsToMatrixFromTensor(d_weights_);
	return d_weights_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution1D::GetDBiases() const
{
	d_biases_cache_ = d_biases_.transpose();
	return d_biases_cache_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetWeightMomentums() const
{
	weight_momentums_matrix_cache_ = WeightsToMatrixFromTensor(weight_momentums_);
	return weight_momentums_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution1D::GetBiasMomentums() const
{
	bias_momentums_cache_ = bias_momentums_.transpose();
	return bias_momentums_cache_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Convolution1D::GetWeightCaches() const
{
	weight_caches_matrix_cache_ = WeightsToMatrixFromTensor(weight_caches_);
	return weight_caches_matrix_cache_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution1D::GetBiasCaches() const
{
	bias_caches_cache_ = bias_caches_.transpose();
	return bias_caches_cache_;
}

void NEURAL_NETWORK::Convolution1D::SetWeightMomentums(const Eigen::MatrixXd& weight_momentums)
{
	Eigen::Tensor<double, 3> tmp;
	MatrixToWeightsTensor(weight_momentums, tmp);
	SetWeightMomentumsTensor(tmp);
}

void NEURAL_NETWORK::Convolution1D::SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums)
{
	SetBiasMomentumsTensor(bias_momentums.transpose());
}

void NEURAL_NETWORK::Convolution1D::SetWeightCaches(const Eigen::MatrixXd& weight_caches)
{
	Eigen::Tensor<double, 3> tmp;
	MatrixToWeightsTensor(weight_caches, tmp);
	SetWeightCachesTensor(tmp);
}

void NEURAL_NETWORK::Convolution1D::SetBiasCaches(const Eigen::RowVectorXd& bias_caches)
{
	SetBiasCachesTensor(bias_caches.transpose());
}

void NEURAL_NETWORK::Convolution1D::UpdateWeights(Eigen::MatrixXd& weight_update)
{
	Eigen::Tensor<double, 3> update_tensor;
	MatrixToWeightsTensor(weight_update, update_tensor);
	UpdateWeightsTensor(update_tensor);
	weight_update = WeightsToMatrixFromTensor(update_tensor);
}

void NEURAL_NETWORK::Convolution1D::UpdateWeightsCache(Eigen::MatrixXd& weight_update)
{
	Eigen::Tensor<double, 3> update_tensor;
	MatrixToWeightsTensor(weight_update, update_tensor);
	UpdateWeightsCacheTensor(update_tensor);
	weight_update = WeightsToMatrixFromTensor(update_tensor);
}

void NEURAL_NETWORK::Convolution1D::UpdateBiases(Eigen::RowVectorXd& bias_update)
{
	Eigen::VectorXd vec = bias_update.transpose();
	UpdateBiasesTensor(vec);
	bias_update = vec.transpose();
}

void NEURAL_NETWORK::Convolution1D::UpdateBiasesCache(Eigen::RowVectorXd& bias_update)
{
	Eigen::VectorXd vec = bias_update.transpose();
	UpdateBiasesCacheTensor(vec);
	bias_update = vec.transpose();
}

int NEURAL_NETWORK::Convolution1D::GetNumberOfFilters() const 
{
	return number_of_filters_;
}

int NEURAL_NETWORK::Convolution1D::GetFilterLength() const 
{
	return filter_length_;
}

int NEURAL_NETWORK::Convolution1D::GetInputLength() const 
{
	return input_length_;
}

int NEURAL_NETWORK::Convolution1D::GetInputChannels() const 
{
	return input_channels_;
}

int NEURAL_NETWORK::Convolution1D::GetPadding() const 
{
	return padding_;
}

int NEURAL_NETWORK::Convolution1D::GetStride() const 
{
	return stride_;
}

const Eigen::Tensor<double, 3>& NEURAL_NETWORK::Convolution1D::GetWeightsTensor() const
{
	return weights_;
}

const Eigen::VectorXd& NEURAL_NETWORK::Convolution1D::GetBiasesVector() const
{
	return biases_;
}

void NEURAL_NETWORK::Convolution1D::InputMatrixToTensor(const Eigen::MatrixXd& matrix,
										int batch_size,
										int length,
										int channels)
{
	inputs_.resize(batch_size, length, channels);
	for (int b = 0; b < batch_size; b++)
	{
		for (int t = 0; t < length; t++)
		{
			for (int c = 0; c < channels; c++)
			{
				const int col = t * channels + c;
				inputs_(b, t, c) = matrix(b, col);
			}
		}
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution1D::InputTensorToMatrix(const Eigen::Tensor<double, 3>& tensor) const
{
	const int batch_size = tensor.dimension(0);
	const int length = tensor.dimension(1);
	const int channels = tensor.dimension(2);

	Eigen::MatrixXd mat(batch_size, length * channels);
	for (int b = 0; b < batch_size; b++)
	{
		for (int t = 0; t < length; t++)
		{
			for (int c = 0; c < channels; c++)
			{
				const int col = t * channels + c;
				mat(b, col) = tensor(b, t, c);
			}
		}
	}
	return mat;
}

void NEURAL_NETWORK::Convolution1D::WeightsToTensor(const Eigen::MatrixXd& weights_matrix)
{
	int filter_length = weights_.dimension(0);
	int input_channels = weights_.dimension(1);
	int num_filters = weights_.dimension(2);

	if (d_weights_.size() == 0)
	{
		d_weights_ = Eigen::Tensor<double, 3>(filter_length, input_channels, num_filters);
	}

	auto expected_size = filter_length * input_channels * num_filters;

	if (weights_matrix.rows() * weights_matrix.cols() == expected_size &&
		weights_matrix.rows() == num_filters &&
		weights_matrix.cols() == filter_length * input_channels)
	{
		Eigen::Map<Eigen::MatrixXd> weights_as_matrix(d_weights_.data(),
													  num_filters,
													  filter_length * input_channels);
		weights_as_matrix = weights_matrix;
	}
	else
	{
		for (int f = 0; f < num_filters; f++)
		{
			int col = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int k = 0; k < filter_length; k++)
				{
					d_weights_(k, c, f) = weights_matrix(f, col++);
				}
			}
		}
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution1D::WeightsToMatrix() const
{
	return WeightsToMatrixFromTensor(weights_);
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution1D::WeightsToMatrixFromTensor(const Eigen::Tensor<double, 3>& tensor) const
{
	int filter_length = tensor.dimension(0);
	int input_channels = tensor.dimension(1);
	int num_filters = tensor.dimension(2);

	auto expected_size = filter_length * input_channels * num_filters;

	if (tensor.size() == expected_size)
	{
		Eigen::Map<const Eigen::MatrixXd> tensor_map(tensor.data(),
													 num_filters,
													 filter_length * input_channels);
		return tensor_map;
	}
	else
	{
		Eigen::MatrixXd matrix(num_filters, filter_length * input_channels);

		for (int f = 0; f < num_filters; f++)
		{
			int col = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int k = 0; k < filter_length; k++)
				{
					matrix(f, col++) = tensor(k, c, f);
				}
			}
		}

		return matrix;
	}
}

void NEURAL_NETWORK::Convolution1D::MatrixToWeightsTensor(const Eigen::MatrixXd& matrix,
										  Eigen::Tensor<double, 3>& tensor) const
{
	int filter_length = weights_.dimension(0);
	int input_channels = weights_.dimension(1);
	int num_filters = weights_.dimension(2);

	if (tensor.size() == 0)
	{
		tensor = Eigen::Tensor<double, 3>(filter_length, input_channels, num_filters);
	}

	auto expected_size = filter_length * input_channels * num_filters;

	if (matrix.rows() * matrix.cols() == expected_size &&
		matrix.rows() == num_filters &&
		matrix.cols() == filter_length * input_channels)
	{
		Eigen::Map<Eigen::MatrixXd> tensor_as_matrix(tensor.data(),
													 num_filters,
													 filter_length * input_channels);
		tensor_as_matrix = matrix;
	}
	else
	{
		for (int f = 0; f < num_filters; f++)
		{
			int col = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int k = 0; k < filter_length; k++)
				{
					tensor(k, c, f) = matrix(f, col++);
				}
			}
		}
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Convolution1D::im2col(const Eigen::Tensor<double, 3>& input_tensor,
													  int filter_length, int pad_l, int stride)
{
	int batch_size = input_tensor.dimension(0);
	int input_length = input_tensor.dimension(1);
	int channels = input_tensor.dimension(2);

	int effective_len = input_length + 2 * pad_l;
	int out_length = 1 + (effective_len - filter_length) / stride;

	int col_height = filter_length * channels;
	int col_width = out_length * batch_size;

	Eigen::MatrixXd im2col_matrix(col_height, col_width);
	int col_index = 0;

	for (int b = 0; b < batch_size; b++)
	{
		for (int out_t = 0; out_t < out_length; out_t++)
		{
			int row_index = 0;
			for (int c = 0; c < channels; c++)
			{
				for (int k = 0; k < filter_length; k++)
				{
					int t = out_t * stride - pad_l + k;
					double value = 0.0;

					if (t >= 0 && t < input_length)
					{
						value = input_tensor(b, t, c);
					}

					im2col_matrix(row_index, col_index) = value;
					row_index++;
				}
			}
			col_index++;
		}
	}

	return im2col_matrix;
}

void NEURAL_NETWORK::Convolution1D::col2im(const Eigen::MatrixXd& col_matrix,
						   int batch_size,
						   int input_length,
						   int input_channels,
						   int filter_length,
						   int pad_l,
						   int stride)
{
	if (d_input_tensor_.size() == 0 ||
		d_input_tensor_.dimension(0) != batch_size ||
		d_input_tensor_.dimension(1) != input_length ||
		d_input_tensor_.dimension(2) != input_channels)
	{
		d_input_tensor_ = Eigen::Tensor<double, 3>(batch_size, input_length, input_channels);
	}

	d_input_tensor_.setZero();

	int effective_len = input_length + 2 * pad_l;
	int out_length = 1 + (effective_len - filter_length) / stride;

	int col_index = 0;

	for (int b = 0; b < batch_size; b++)
	{
		for (int out_t = 0; out_t < out_length; out_t++)
		{
			int row_index = 0;
			for (int c = 0; c < input_channels; c++)
			{
				for (int k = 0; k < filter_length; k++)
				{
					int t = out_t * stride - pad_l + k;

					if (t >= 0 && t < input_length)
					{
						d_input_tensor_(b, t, c) += col_matrix(row_index, col_index);
					}

					row_index++;
				}
			}
			col_index++;
		}
	}
}


void NEURAL_NETWORK::Convolution1D::SetWeightMomentumsTensor(const Eigen::Tensor<double, 3>& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::Convolution1D::SetBiasMomentumsTensor(const Eigen::VectorXd& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::Convolution1D::SetWeightCachesTensor(const Eigen::Tensor<double, 3>& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::Convolution1D::SetBiasCachesTensor(const Eigen::VectorXd& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::Convolution1D::UpdateWeightsTensor(Eigen::Tensor<double, 3>& weight_update)
{
	weights_ = weights_ + weight_update;
}

void NEURAL_NETWORK::Convolution1D::UpdateWeightsCacheTensor(Eigen::Tensor<double, 3>& weight_update)
{
	weight_caches_ = weight_caches_ + weight_update;
}

void NEURAL_NETWORK::Convolution1D::UpdateBiasesTensor(Eigen::VectorXd& bias_update)
{
	biases_ = biases_ + bias_update;
}

void NEURAL_NETWORK::Convolution1D::UpdateBiasesCacheTensor(Eigen::VectorXd& bias_update)
{
	bias_caches_ = bias_caches_ + bias_update;
}
