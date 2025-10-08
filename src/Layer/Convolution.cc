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

	Eigen::Tensor<double, 4> input_tensor = MatrixToTensor(inputs, batch_size, input_height_, input_width_, input_channels);

	Eigen::MatrixXd im2col_matrix = im2col(input_tensor,
										   filter_height, filter_width,
										   pad_height, pad_width,
										   stride_height_, stride_width_);

	const Eigen::MatrixXd filter_matrix = WeightsToMatrix();

	Eigen::MatrixXd result = filter_matrix * im2col_matrix;

	result.colwise() += biases_; 

	int output_height = ((input_height_ + (2 * pad_height) - filter_height) / stride_height_) + 1;
	int output_width =  ((input_width_ + (2 * pad_width) - filter_width) / stride_width_) + 1;

	tensor_output_ = Eigen::Tensor<double, 4>(batch_size, output_height, output_width, num_filters);

	Eigen::MatrixXd result_transposed = result.transpose();

	int total_spatial = output_height * output_width * batch_size;
	Eigen::Map<Eigen::MatrixXd> tensor_map(tensor_output_.data(), total_spatial, num_filters);

	tensor_map = result_transposed;

	output_ = TensorToMatrix(tensor_output_);

}

void NEURAL_NETWORK::Convolution::backward(const Eigen::MatrixXd& dvalues)
{
	d_input_ = dvalues;
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Convolution::GetWeights() const //maybe weights in the n-th filter?
{
	return weights_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::Convolution::GetBiases() const
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

	Eigen::MatrixXd filter_matrix(num_filters, filter_height * filter_width * input_channels);

	for (int f = 0; f < num_filters; f++)
	{
		int col = 0;
		for (int c = 0; c < input_channels; c++)
		{
			for (int h = 0; h < filter_height; h++)
			{
				for (int w = 0; w < filter_width; w++)
				{
					filter_matrix(f, col++) = weights_(h, w, c, f);
				}
			}
		}
	}

	return filter_matrix;
}




/*
 What Eigen::Map Does:

  Eigen::Map creates a view of existing memory as an Eigen matrix without copying data.

  Eigen::Map<Eigen::MatrixXd> tensor_map(tensor_output_.data(), total_spatial,num_filters);
  //                                     ^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^  ^^^^^^^^^^^
  //                                     memory pointer        rows           columns

  Step by Step:

  1. Memory Layout of tensor_output_:

  tensor_output_ = Eigen::Tensor<double, 4>(batch_size, output_height, output_width, num_filters);
  // Shape: [batch_size=2, output_height=3, output_width=3, num_filters=4]
  // Memory: [b0h0w0f0, b0h0w0f1, b0h0w0f2, b0h0w0f3, b0h0w1f0, b0h0w1f1, ...]
  //         |<-- 4 filters -->|          |<-- 4 filters -->|

  2. Map Interpretation:

  int total_spatial = output_height * output_width * batch_size; // 3×3×2 = 18
  Eigen::Map<Eigen::MatrixXd> tensor_map(tensor_output_.data(), 18, 4);

  The map sees the same memory as an 18×4 matrix:
  Row 0: [b0h0w0f0, b0h0w0f1, b0h0w0f2, b0h0w0f3]  ← All filters for position (b0,h0,w0)
  Row 1: [b0h0w1f0, b0h0w1f1, b0h0w1f2, b0h0w1f3]  ← All filters for position (b0,h0,w1)
  Row 2: [b0h0w2f0, b0h0w2f1, b0h0w2f2, b0h0w2f3]  ← All filters for position (b0,h0,w2)
  ...
  Row 17:[b1h2w2f0, b1h2w2f1, b1h2w2f2, b1h2w2f3]  ← All filters for position (b1,h2,w2)

  3. Assignment:

  tensor_map = result_transposed;
  // result_transposed is also [18×4]
  // This copies values directly into tensor_output_'s memory

  Why This Works:

  1. Same memory layout: Eigen tensors store data in row-major order (last
  dimension varies fastest)
  2. Compatible shapes: Both the map and result_transposed are 18×4
  3. Efficient: No loops, uses optimized memory copy operations
  4. Zero-copy view: tensor_map doesn't allocate new memory, just reinterprets
  existing memory

  Example with Small Numbers:

  // If output is 2×2×1×2 (batch=2, height=2, width=2, filters=2):
  // tensor_output_ memory: [b0h0w0f0, b0h0w0f1, b0h0w1f0, b0h0w1f1, b0h1w0f0,
  b0h1w0f1, b0h1w1f0, b0h1w1f1, ...]

  // Map sees it as 8×2 matrix:
  // Row 0: [b0h0w0f0, b0h0w0f1]
  // Row 1: [b0h0w1f0, b0h0w1f1]
  // Row 2: [b0h1w0f0, b0h1w0f1]
  // Row 3: [b0h1w1f0, b0h1w1f1]
  // Row 4: [b1h0w0f0, b1h0w0f1]
  // ...

  This is a very elegant way to avoid explicit loops while maintaining memory
  efficiency!
*/