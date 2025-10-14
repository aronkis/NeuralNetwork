#include "BatchNormalization.h"

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