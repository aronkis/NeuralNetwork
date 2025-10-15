#include "ActivationReLU.h"
#include "TensorUtils.h"

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::MatrixXd& inputs, 
											 bool training)
{
	inputs_ = inputs;
	output_ = inputs.cwiseMax(0.0);
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::MatrixXd& d_values)
{
	d_inputs_ = d_values;
	d_inputs_.array() *= (inputs_.array() > 0.0).cast<double>().array();
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::ActivationReLU::GetDInput() const
{
	return d_inputs_;
}

void NEURAL_NETWORK::ActivationReLU::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_inputs_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::ActivationReLU::predictions() const
{
	return output_;
}

// Tensor interface implementations
bool NEURAL_NETWORK::ActivationReLU::SupportsTensorInterface() const
{
	return true;
}

void NEURAL_NETWORK::ActivationReLU::forward(const Eigen::Tensor<double, 4>& inputs, bool training)
{
	tensor_inputs_ = inputs;
	tensor_output_ = tensor_inputs_.cwiseMax(0.0);

	// Manual conversion for backward compatibility - no TensorUtils call count
	int batch_size = inputs.dimension(0);
	int height = inputs.dimension(1);
	int width = inputs.dimension(2);
	int channels = inputs.dimension(3);
	int num_features = height * width * channels;

	Eigen::MatrixXd matrix_input(batch_size, num_features);
	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					matrix_input(b, idx) = inputs(b, h, w, c);
				}
			}
		}
	}
	forward(matrix_input, training);
}

void NEURAL_NETWORK::ActivationReLU::backward(const Eigen::Tensor<double, 4>& dvalues)
{
	tensor_d_inputs_ = dvalues;
	// Apply ReLU derivative: gradient * (input > 0)
	tensor_d_inputs_ = tensor_d_inputs_ * (tensor_inputs_ > tensor_inputs_.constant(0.0)).cast<double>();

	// Manual conversion for backward compatibility - no TensorUtils call count
	int batch_size = dvalues.dimension(0);
	int height = dvalues.dimension(1);
	int width = dvalues.dimension(2);
	int channels = dvalues.dimension(3);
	int num_features = height * width * channels;

	Eigen::MatrixXd matrix_dvalues(batch_size, num_features);
	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					matrix_dvalues(b, idx) = dvalues(b, h, w, c);
				}
			}
		}
	}
	backward(matrix_dvalues);
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::ActivationReLU::GetTensorOutput() const
{
	return tensor_output_;
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::ActivationReLU::GetTensorDInput() const
{
	return tensor_d_inputs_;
}

void NEURAL_NETWORK::ActivationReLU::SetTensorDInput(const Eigen::Tensor<double, 4>& dinput)
{
	tensor_d_inputs_ = dinput;
	// Manual conversion for backward compatibility - no TensorUtils call count
	int batch_size = dinput.dimension(0);
	int height = dinput.dimension(1);
	int width = dinput.dimension(2);
	int channels = dinput.dimension(3);
	int num_features = height * width * channels;

	if (d_inputs_.rows() != batch_size || d_inputs_.cols() != num_features) {
		d_inputs_ = Eigen::MatrixXd(batch_size, num_features);
	}

	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					d_inputs_(b, idx) = dinput(b, h, w, c);
				}
			}
		}
	}
}
