#include <random>
#include <cmath>
#include "LayerDense.h"

NEURAL_NETWORK::LayerDense::LayerDense(int n_inputs, int n_neurons,
									   double weight_regularizer_l1, 
									   double weight_regularizer_l2,
									   double bias_regularizer_l1, 
									   double bias_regularizer_l2)
{
	std::mt19937 gen(0);
	// He normal initialization
	const double he_std = std::sqrt(2.0 / static_cast<double>(n_inputs));
	std::normal_distribution<> he_dist(0.0, he_std);
	weights_ = Eigen::MatrixXd::NullaryExpr(n_inputs, n_neurons, [&]() { return he_dist(gen); });

	weight_regularizer_l1_ = weight_regularizer_l1;
	weight_regularizer_l2_ = weight_regularizer_l2;
	bias_regularizer_l1_ = bias_regularizer_l1;
	bias_regularizer_l2_ = bias_regularizer_l2;

	biases_ = Eigen::RowVectorXd::Zero(n_neurons);
}

void NEURAL_NETWORK::LayerDense::forward(const Eigen::MatrixXd& inputs, 
										 bool training)
{
	inputs_ = inputs;
	output_.resize(inputs.rows(), weights_.cols());
	output_.noalias() = inputs * weights_;
	output_.rowwise() += biases_;
}

void NEURAL_NETWORK::LayerDense::backward(const Eigen::MatrixXd& d_values)
{
	d_weights_.resize(weights_.rows(), weights_.cols());
	d_weights_.noalias() = inputs_.transpose() * d_values;
	d_biases_ = d_values.colwise().sum();
	
	if (weight_regularizer_l1_ > 0)
	{
		d_weights_.array() += weight_regularizer_l1_ * weights_.array().sign();
	}

	if (weight_regularizer_l2_ > 0)
	{
		d_weights_.array() += 2 * weight_regularizer_l2_ * weights_.array();
	}

	if (bias_regularizer_l1_ > 0)
	{
		d_biases_.array() += bias_regularizer_l1_ * biases_.array().sign();
	}

	if (bias_regularizer_l2_ > 0)
	{
		d_biases_.array() += 2 * bias_regularizer_l2_ * biases_.array();
	}

	d_inputs_.resize(d_values.rows(), weights_.rows());
	d_inputs_.noalias() = d_values * weights_.transpose();
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeights() const
{
	return weights_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiases() const
{
	return biases_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetOutput() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetDInput() const
{
	return d_inputs_;
}

Eigen::MatrixXd NEURAL_NETWORK::LayerDense::predictions() const
{
	return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetDWeights() const
{
	return d_weights_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetDBiases() const
{
	return d_biases_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeightMomentums() const
{
	return weight_momentums_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiasMomentums() const
{
	return bias_momentums_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerDense::GetWeightCaches() const
{
	return weight_caches_;
}

const Eigen::RowVectorXd& NEURAL_NETWORK::LayerDense::GetBiasCaches() const
{
	return bias_caches_;
}

double NEURAL_NETWORK::LayerDense::GetWeightRegularizerL1() const
{
	return weight_regularizer_l1_;
}

double NEURAL_NETWORK::LayerDense::GetWeightRegularizerL2() const
{
	return weight_regularizer_l2_;
}

double NEURAL_NETWORK::LayerDense::GetBiasRegularizerL1() const
{
	return bias_regularizer_l1_;
}

double NEURAL_NETWORK::LayerDense::GetBiasRegularizerL2() const
{
	return bias_regularizer_l2_;
}

std::pair<const Eigen::MatrixXd&, const Eigen::RowVectorXd&> NEURAL_NETWORK::LayerDense::GetParameters() const
{
    return { weights_, biases_ };
}

void NEURAL_NETWORK::LayerDense::SetDInput(const Eigen::MatrixXd& dinput)  
{ 
	d_inputs_ = dinput; 
}


void NEURAL_NETWORK::LayerDense::SetWeightMomentums(const Eigen::MatrixXd& weight_momentums)
{
	weight_momentums_ = weight_momentums;
}

void NEURAL_NETWORK::LayerDense::SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums)
{
	bias_momentums_ = bias_momentums;
}

void NEURAL_NETWORK::LayerDense::SetWeightCaches(const Eigen::MatrixXd& weight_caches)
{
	weight_caches_ = weight_caches;
}

void NEURAL_NETWORK::LayerDense::SetBiasCaches(const Eigen::RowVectorXd& bias_caches)
{
	bias_caches_ = bias_caches;
}

void NEURAL_NETWORK::LayerDense::UpdateWeights(Eigen::MatrixXd& weight_update)
{
	weights_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateWeightsCache(Eigen::MatrixXd& weight_update)
{
    weight_caches_ += weight_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiases(Eigen::RowVectorXd& bias_update)
{
	biases_ += bias_update;
}

void NEURAL_NETWORK::LayerDense::UpdateBiasesCache(Eigen::RowVectorXd& bias_update)
{
    bias_caches_ += bias_update;
}

void NEURAL_NETWORK::LayerDense::SetParameters(const Eigen::MatrixXd& weights, 
											   const Eigen::RowVectorXd& biases)
{
	weights_ = weights;
	biases_ = biases;
}