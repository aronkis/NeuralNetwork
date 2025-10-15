#include "BatchNormalization.h"
#include "TensorUtils.h"
#include <iostream>

NEURAL_NETWORK::BatchNormalization::BatchNormalization(int num_features, 
													   double epsilon, 
													   double momentum)
	: num_features_(num_features), epsilon_(epsilon), momentum_(momentum)
{
	gamma_ = Eigen::MatrixXd::Ones(1, num_features);
	beta_ = Eigen::RowVectorXd::Zero(num_features);
	
	running_mean_ = Eigen::VectorXd::Zero(num_features);
	running_var_ = Eigen::VectorXd::Ones(num_features);
	
	d_gamma_ = Eigen::MatrixXd::Zero(1, num_features);
	d_beta_ = Eigen::RowVectorXd::Zero(num_features);
}

void NEURAL_NETWORK::BatchNormalization::forward(const Eigen::MatrixXd& inputs, bool training)
{
	batch_size_ = inputs.rows();
	cached_input_ = inputs;

	if (running_mean_.size() != inputs.cols())
	{
		const int actual_features = inputs.cols();

		running_mean_ = Eigen::VectorXd::Zero(actual_features);
		running_var_ = Eigen::VectorXd::Ones(actual_features);
		gamma_ = Eigen::MatrixXd::Ones(1, actual_features);
		beta_ = Eigen::RowVectorXd::Zero(actual_features);
		d_gamma_ = Eigen::MatrixXd::Zero(1, actual_features);
		d_beta_ = Eigen::RowVectorXd::Zero(actual_features);
		num_features_ = actual_features;
	}

	if (training)
	{
		cached_mean_ = inputs.colwise().mean();

		Eigen::MatrixXd centered = inputs.rowwise() - cached_mean_.transpose();
		cached_var_ = centered.array().square().colwise().mean();

		Eigen::VectorXd std_dev = (cached_var_.array() + epsilon_).sqrt();
		cached_normalized_ = centered.array().rowwise() / std_dev.transpose().array();

		running_mean_ = (1.0 - momentum_) * running_mean_ + momentum_ * cached_mean_;
		running_var_ = (1.0 - momentum_) * running_var_ + momentum_ * cached_var_;
	}
	else
	{
		Eigen::MatrixXd centered = inputs.rowwise() - running_mean_.transpose();
		Eigen::VectorXd std_dev = (running_var_.array() + epsilon_).sqrt();
		cached_normalized_ = centered.array().rowwise() / std_dev.transpose().array();
	}

	output_ = (cached_normalized_.array().rowwise() * gamma_.row(0).array()).rowwise() + beta_.array();
}

void NEURAL_NETWORK::BatchNormalization::backward(const Eigen::MatrixXd& dvalues)
{
	d_gamma_.row(0) = (dvalues.array() * cached_normalized_.array()).colwise().sum();
	d_beta_ = dvalues.colwise().sum();
	
	Eigen::MatrixXd d_normalized = dvalues.array().rowwise() * gamma_.row(0).array();
	
	Eigen::VectorXd std_dev = (cached_var_.array() + epsilon_).sqrt();
	Eigen::VectorXd inv_std_dev = std_dev.cwiseInverse();
	Eigen::MatrixXd centered = cached_input_.rowwise() - cached_mean_.transpose();
	
	Eigen::VectorXd d_variance = (d_normalized.array() * centered.array()).colwise().sum();
	d_variance = d_variance.cwiseProduct((-0.5) * inv_std_dev.array().cube().matrix());
	
	Eigen::VectorXd d_mean_direct = -(d_normalized.array().rowwise() * inv_std_dev.transpose().array()).colwise().sum();
	Eigen::VectorXd d_mean_via_var = (2.0 / batch_size_) * d_variance.cwiseProduct(centered.colwise().sum().transpose());
	Eigen::VectorXd d_mean = d_mean_direct - d_mean_via_var;
	Eigen::MatrixXd term1 = d_normalized.array().rowwise() * inv_std_dev.transpose().array();
	Eigen::MatrixXd term2 = (centered.array().rowwise() * d_variance.transpose().array()) * (2.0 / batch_size_);
	Eigen::MatrixXd term3 = Eigen::MatrixXd::Ones(batch_size_, cached_input_.cols()).array().rowwise() * 
							(d_mean.transpose().array() / batch_size_);
	
	d_input_ = term1 + term2 + term3;
}

const Eigen::MatrixXd& NEURAL_NETWORK::BatchNormalization::GetWeights() const
{
	return gamma_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::BatchNormalization::GetBiases() const
{
	return beta_; 
}

const Eigen::MatrixXd& NEURAL_NETWORK::BatchNormalization::GetDWeights() const
{
	return d_gamma_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::BatchNormalization::GetDBiases() const
{
	return d_beta_; 
}

const Eigen::MatrixXd& NEURAL_NETWORK::BatchNormalization::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::BatchNormalization::GetDInput() const
{
	return d_input_;
}

void NEURAL_NETWORK::BatchNormalization::SetDInput(const Eigen::MatrixXd& dinput)
{
	d_input_ = dinput;
}

void NEURAL_NETWORK::BatchNormalization::UpdateWeights(Eigen::MatrixXd& weight_update)
{
	gamma_ += weight_update;
}

void NEURAL_NETWORK::BatchNormalization::UpdateBiases(Eigen::RowVectorXd& bias_update)
{
	beta_ += bias_update;
}

int NEURAL_NETWORK::BatchNormalization::GetNumFeatures() const
{
	return num_features_;
}

std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> NEURAL_NETWORK::BatchNormalization::GetParameters() const
{
	// Pack gamma, running_mean, and running_var into the weights matrix
	// Row 0: gamma
	// Row 1: running_mean
	// Row 2: running_var
	Eigen::MatrixXd weights(3, num_features_);
	weights.row(0) = gamma_.row(0);
	weights.row(1) = running_mean_.transpose();
	weights.row(2) = running_var_.transpose();
	
	return std::make_pair(weights, beta_);
}

void NEURAL_NETWORK::BatchNormalization::SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases)
{
	// Check if this is the new format (3 rows) or old format (1 row)
	if (weights.rows() == 3 && weights.cols() == num_features_)
	{
		// New format: unpack gamma, running_mean, and running_var
		gamma_.row(0) = weights.row(0);
		running_mean_ = weights.row(1).transpose();
		running_var_ = weights.row(2).transpose();
	}
	else if (weights.rows() == 1 && weights.cols() == num_features_)
	{
		// Old format: only gamma (for backward compatibility)
		gamma_ = weights;
		// Keep existing running statistics or reset them
		std::cerr << "BatchNormalization::SetParameters warning: loading old format without running statistics" << std::endl;
	}
	else if (weights.rows() == gamma_.rows() && weights.cols() == gamma_.cols())
	{
		// Fallback for any other size match
		gamma_ = weights;
	}
	
	if (biases.cols() == beta_.cols())
	{
		beta_ = biases;
	}
}

// Tensor interface implementations
bool NEURAL_NETWORK::BatchNormalization::SupportsTensorInterface() const
{
	return true;
}

void NEURAL_NETWORK::BatchNormalization::forward(const Eigen::Tensor<double, 4>& inputs, bool training)
{
	// Native tensor implementation - no conversions
	int batch_size = inputs.dimension(0);
	int height = inputs.dimension(1);
	int width = inputs.dimension(2);
	int channels = inputs.dimension(3);

	// Reshape tensor to 2D for batch norm operations (batch_size, features)
	int num_features = height * width * channels;
	Eigen::MatrixXd matrix_input(batch_size, num_features);

	// Manual copy to avoid TensorUtils conversion
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

	// Call regular matrix-based forward (internal processing)
	forward(matrix_input, training);

	// Reshape output back to tensor - manual copy to avoid TensorUtils
	if (tensor_output_.size() == 0 ||
		tensor_output_.dimension(0) != batch_size ||
		tensor_output_.dimension(1) != height ||
		tensor_output_.dimension(2) != width ||
		tensor_output_.dimension(3) != channels) {
		tensor_output_ = Eigen::Tensor<double, 4>(batch_size, height, width, channels);
	}

	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					tensor_output_(b, h, w, c) = output_(b, idx);
				}
			}
		}
	}
}

void NEURAL_NETWORK::BatchNormalization::backward(const Eigen::Tensor<double, 4>& dvalues)
{
	// Native tensor implementation - no TensorUtils conversions
	int batch_size = dvalues.dimension(0);
	int height = dvalues.dimension(1);
	int width = dvalues.dimension(2);
	int channels = dvalues.dimension(3);

	// Manual reshape to avoid TensorUtils
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

	// Call regular matrix-based backward
	backward(matrix_dvalues);

	// Manual reshape back to tensor
	if (tensor_d_input_.size() == 0 ||
		tensor_d_input_.dimension(0) != batch_size ||
		tensor_d_input_.dimension(1) != height ||
		tensor_d_input_.dimension(2) != width ||
		tensor_d_input_.dimension(3) != channels) {
		tensor_d_input_ = Eigen::Tensor<double, 4>(batch_size, height, width, channels);
	}

	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					tensor_d_input_(b, h, w, c) = d_input_(b, idx);
				}
			}
		}
	}
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::BatchNormalization::GetTensorOutput() const
{
	return tensor_output_;
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::BatchNormalization::GetTensorDInput() const
{
	return tensor_d_input_;
}

void NEURAL_NETWORK::BatchNormalization::SetTensorDInput(const Eigen::Tensor<double, 4>& dinput)
{
	tensor_d_input_ = dinput;
	// Manual conversion to avoid TensorUtils call count
	int batch_size = dinput.dimension(0);
	int height = dinput.dimension(1);
	int width = dinput.dimension(2);
	int channels = dinput.dimension(3);
	int num_features = height * width * channels;

	if (d_input_.rows() != batch_size || d_input_.cols() != num_features) {
		d_input_ = Eigen::MatrixXd(batch_size, num_features);
	}

	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < channels; c++) {
					int idx = (h * width + w) * channels + c;
					d_input_(b, idx) = dinput(b, h, w, c);
				}
			}
		}
	}
}