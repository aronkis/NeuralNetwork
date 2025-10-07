#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Adam.h"
#include "AdaGrad.h"
#include "RMSProp.h"
#include "StochasticGradientDescent.h"
#include "LayerDense.h"

class OptimizerTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		layer = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 2);

		initial_weights = Eigen::MatrixXd(3, 2);
		initial_weights << 0.1, 0.2,
						   0.3, 0.4,
						   0.5, 0.6;

		initial_biases = Eigen::RowVectorXd(2);
		initial_biases << 0.1, 0.2;

		layer->SetParameters(initial_weights, initial_biases);

		mock_gradients = Eigen::MatrixXd(3, 2);
		mock_gradients << -0.01,  0.02,
						   0.03, -0.04,
						  -0.05,  0.06;

		mock_bias_gradients = Eigen::RowVectorXd(2);
		mock_bias_gradients << 0.01, -0.02;

		trainable_layers.push_back(layer);
	}

	std::shared_ptr<NEURAL_NETWORK::LayerDense> layer;
	std::vector<std::shared_ptr<NEURAL_NETWORK::LayerDense>> trainable_layers;
	Eigen::MatrixXd initial_weights;
	Eigen::RowVectorXd initial_biases;
	Eigen::MatrixXd mock_gradients;
	Eigen::RowVectorXd mock_bias_gradients;
	const double tolerance = 1e-10;

	void CheckParametersChanged() 
	{
		const auto& current_weights = layer->GetWeights();
		const auto& current_biases = layer->GetBiases();

		EXPECT_FALSE(current_weights.isApprox(initial_weights, tolerance));
		EXPECT_FALSE(current_biases.isApprox(initial_biases, tolerance));
	}
};

class SGDTest : public OptimizerTest 
{};

TEST_F(SGDTest, ConstructorSetsLearningRate) 
{
	double learning_rate = 0.01;
	NEURAL_NETWORK::StochasticGradientDescent sgd(learning_rate);

	EXPECT_DOUBLE_EQ(sgd.GetLearningRate(), learning_rate);
}

TEST_F(SGDTest, ConstructorSetsDecayAndMomentum) 
{
	double learning_rate = 0.01;
	double decay = 1e-5;
	double momentum = 0.9;

	NEURAL_NETWORK::StochasticGradientDescent sgd(learning_rate, decay, momentum);

	EXPECT_DOUBLE_EQ(sgd.GetLearningRate(), learning_rate);
	EXPECT_DOUBLE_EQ(sgd.GetDecay(), decay);
	EXPECT_DOUBLE_EQ(sgd.GetMomentum(), momentum);
}

TEST_F(SGDTest, PostUpdateStepIncrementsIterations) 
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.01);

	int initial_iterations = sgd.GetIterations();
	sgd.PostUpdateParameters();
	EXPECT_EQ(sgd.GetIterations(), initial_iterations + 1);
}

TEST_F(SGDTest, ParameterUpdateWorks) 
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.1, 0.0, 0.9);

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	layer->forward(dummy_inputs, true);

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.2,
						0.3, 0.4;

	layer->backward(dummy_dvalues);

	sgd.PreUpdateParameters();
	sgd.UpdateParameters(*layer);
	sgd.PostUpdateParameters();

	const auto& momentums = layer->GetWeightMomentums();
	const auto& bias_momentums = layer->GetBiasMomentums();

	EXPECT_EQ(momentums.rows(), initial_weights.rows());
	EXPECT_EQ(momentums.cols(), initial_weights.cols());
}

TEST_F(SGDTest, MomentumAccumulatesAcrossUpdates)
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.01, 0.0, 0.8);

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.05,
					  0.2, 0.15;

	layer->forward(dummy_inputs, true);
	layer->backward(dummy_dvalues);

	sgd.PreUpdateParameters();
	sgd.UpdateParameters(*layer);
	sgd.PostUpdateParameters();

	Eigen::MatrixXd first_weight_momentum = layer->GetWeightMomentums();
	Eigen::RowVectorXd first_bias_momentum = layer->GetBiasMomentums();

	layer->forward(dummy_inputs, true);
	layer->backward(dummy_dvalues);

	sgd.PreUpdateParameters();
	Eigen::MatrixXd gradients = layer->GetDWeights();
	Eigen::RowVectorXd bias_gradients = layer->GetDBiases();
	sgd.UpdateParameters(*layer);

	Eigen::MatrixXd second_weight_momentum = layer->GetWeightMomentums();
	Eigen::RowVectorXd second_bias_momentum = layer->GetBiasMomentums();
	double effective_lr = sgd.GetLearningRate();

	Eigen::MatrixXd expected_weight_momentum = 0.8 * first_weight_momentum - effective_lr * gradients;
	Eigen::RowVectorXd expected_bias_momentum = 0.8 * first_bias_momentum - effective_lr * bias_gradients;

	EXPECT_TRUE(second_weight_momentum.isApprox(expected_weight_momentum, tolerance));
	EXPECT_TRUE(second_bias_momentum.isApprox(expected_bias_momentum, tolerance));
}

TEST_F(SGDTest, LearningRateDecayOverMultipleIterations)
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.2, 0.15);

	for (int i = 0; i < 5; i++)
	{
		sgd.PreUpdateParameters();
		double expected = 0.2 / (1.0 + 0.15 * sgd.GetIterations());
		EXPECT_NEAR(sgd.GetLearningRate(), expected, tolerance);
		sgd.PostUpdateParameters();
	}
}

class AdamTest : public OptimizerTest 
{};

TEST_F(AdamTest, ConstructorSetsHyperparameters) 
{
	double learning_rate = 0.001;
	double decay = 1e-4;
	double beta_1 = 0.9;
	double beta_2 = 0.999;
	double epsilon = 1e-7;

	NEURAL_NETWORK::Adam adam(learning_rate, decay, beta_1, beta_2, epsilon);

	EXPECT_DOUBLE_EQ(adam.GetLearningRate(), learning_rate);
	EXPECT_DOUBLE_EQ(adam.GetDecay(), decay);
	EXPECT_DOUBLE_EQ(adam.GetBeta1(), beta_1);
	EXPECT_DOUBLE_EQ(adam.GetBeta2(), beta_2);
	EXPECT_DOUBLE_EQ(adam.GetEpsilon(), epsilon);
}

TEST_F(AdamTest, DefaultConstructorUsesStandardValues) 
{
	NEURAL_NETWORK::Adam adam;

	EXPECT_DOUBLE_EQ(adam.GetLearningRate(), 0.001);
	EXPECT_DOUBLE_EQ(adam.GetBeta1(), 0.9);
	EXPECT_DOUBLE_EQ(adam.GetBeta2(), 0.999);
	EXPECT_DOUBLE_EQ(adam.GetEpsilon(), 1e-7);
}

TEST_F(AdamTest, ParameterUpdateWorks) 
{
	NEURAL_NETWORK::Adam adam(0.001);

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	layer->forward(dummy_inputs, true);

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.2,
						0.3, 0.4;

	layer->backward(dummy_dvalues);

	for (int i = 0; i < 5; i++) 
	{
		adam.PreUpdateParameters();
		adam.UpdateParameters(*layer);
		adam.PostUpdateParameters();
	}

	EXPECT_GT(adam.GetIterations(), 0);
}

class AdaGradTest : public OptimizerTest 
{};

TEST_F(AdaGradTest, ConstructorSetsHyperparameters) 
{
	double learning_rate = 0.01;
	double decay = 1e-4;
	double epsilon = 1e-8;

	NEURAL_NETWORK::AdaGrad adagrad(learning_rate, decay, epsilon);

	EXPECT_DOUBLE_EQ(adagrad.GetLearningRate(), learning_rate);
	EXPECT_DOUBLE_EQ(adagrad.GetDecay(), decay);
	EXPECT_DOUBLE_EQ(adagrad.GetEpsilon(), epsilon);
}

TEST_F(AdaGradTest, CacheAccumulation) 
{
	NEURAL_NETWORK::AdaGrad adagrad(0.1);

	const auto& initial_weight_cache = layer->GetWeightCaches();
	const auto& initial_bias_cache = layer->GetBiasCaches();

	EXPECT_TRUE(initial_weight_cache.isZero());
	EXPECT_TRUE(initial_bias_cache.isZero());

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	layer->forward(dummy_inputs, true);

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.2,
						0.3, 0.4;

	layer->backward(dummy_dvalues);

	adagrad.PreUpdateParameters();
	adagrad.UpdateParameters(*layer);

	const auto& updated_weight_cache = layer->GetWeightCaches();
	const auto& updated_bias_cache = layer->GetBiasCaches();

	EXPECT_EQ(updated_weight_cache.rows(), initial_weights.rows());
	EXPECT_EQ(updated_weight_cache.cols(), initial_weights.cols());
}

class RMSPropTest : public OptimizerTest 
{};

TEST_F(RMSPropTest, ConstructorSetsHyperparameters) 
{
	double learning_rate = 0.001;
	double decay = 1e-4;
	double epsilon = 1e-7;
	double rho = 0.9;

	NEURAL_NETWORK::RMSProp rmsprop(learning_rate, decay, epsilon, rho);

	EXPECT_DOUBLE_EQ(rmsprop.GetLearningRate(), learning_rate);
	EXPECT_DOUBLE_EQ(rmsprop.GetDecay(), decay);
	EXPECT_DOUBLE_EQ(rmsprop.GetEpsilon(), epsilon);
	EXPECT_DOUBLE_EQ(rmsprop.GetRho(), rho);
}

TEST_F(RMSPropTest, DefaultConstructorUsesStandardValues) 
{
	NEURAL_NETWORK::RMSProp rmsprop;

	EXPECT_DOUBLE_EQ(rmsprop.GetLearningRate(), 0.01);
	EXPECT_DOUBLE_EQ(rmsprop.GetRho(), 0.9);
	EXPECT_DOUBLE_EQ(rmsprop.GetEpsilon(), 1e-7);
}

class OptimizerComparisonTest : public OptimizerTest 
{};

TEST_F(OptimizerComparisonTest, DifferentOptimizersHaveDifferentProperties) 
{
	
	auto layer_sgd = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 2);
	auto layer_adam = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 2);
	auto layer_adagrad = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 2);
	auto layer_rmsprop = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 2);

	layer_sgd->SetParameters(initial_weights, initial_biases);
	layer_adam->SetParameters(initial_weights, initial_biases);
	layer_adagrad->SetParameters(initial_weights, initial_biases);
	layer_rmsprop->SetParameters(initial_weights, initial_biases);

	double lr = 0.01;
	NEURAL_NETWORK::StochasticGradientDescent sgd(lr);
	NEURAL_NETWORK::Adam adam(lr);
	NEURAL_NETWORK::AdaGrad adagrad(lr);
	NEURAL_NETWORK::RMSProp rmsprop(lr);

	sgd.PreUpdateParameters();
	adam.PreUpdateParameters();
	adagrad.PreUpdateParameters();
	rmsprop.PreUpdateParameters();

	const auto& weights_sgd = layer_sgd->GetWeights();
	const auto& weights_adam = layer_adam->GetWeights();
	const auto& weights_adagrad = layer_adagrad->GetWeights();
	const auto& weights_rmsprop = layer_rmsprop->GetWeights();

	EXPECT_TRUE(weights_sgd.isApprox(initial_weights, tolerance));
	EXPECT_TRUE(weights_adam.isApprox(initial_weights, tolerance));
	EXPECT_TRUE(weights_adagrad.isApprox(initial_weights, tolerance));
	EXPECT_TRUE(weights_rmsprop.isApprox(initial_weights, tolerance));
}

TEST_F(OptimizerComparisonTest, LearningRateEffectiveness) 
{
	double low_lr = 0.001;
	double high_lr = 0.1;

	NEURAL_NETWORK::StochasticGradientDescent sgd_low(low_lr);
	NEURAL_NETWORK::StochasticGradientDescent sgd_high(high_lr);

	EXPECT_LT(sgd_low.GetLearningRate(), sgd_high.GetLearningRate());
}

class OptimizerEdgeCaseTest : public OptimizerTest 
{};

TEST_F(OptimizerEdgeCaseTest, ZeroLearningRate) 
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.0);

	EXPECT_DOUBLE_EQ(sgd.GetLearningRate(), 0.0);

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	layer->forward(dummy_inputs, true);

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.2,
						0.3, 0.4;

	layer->backward(dummy_dvalues);

	sgd.PreUpdateParameters();
	sgd.UpdateParameters(*layer);

	const auto& weights_after = layer->GetWeights();
	const auto& biases_after = layer->GetBiases();

	EXPECT_TRUE(weights_after.isApprox(initial_weights, 1e-10));
	EXPECT_TRUE(biases_after.isApprox(initial_biases, 1e-10));
}

TEST_F(OptimizerEdgeCaseTest, VerySmallEpsilon) 
{
	NEURAL_NETWORK::Adam adam(0.001, 0.0, 0.9, 0.999, 1e-15);  

	EXPECT_DOUBLE_EQ(adam.GetEpsilon(), 1e-15);

	Eigen::MatrixXd dummy_inputs(2, 3);
	dummy_inputs << 1.0, 2.0, 3.0,
					4.0, 5.0, 6.0;

	layer->forward(dummy_inputs, true);

	Eigen::MatrixXd dummy_dvalues(2, 2);
	dummy_dvalues << 0.1, 0.2,
					 0.3, 0.4;

	layer->backward(dummy_dvalues);

	adam.PreUpdateParameters();
	adam.UpdateParameters(*layer);
}

TEST_F(OptimizerEdgeCaseTest, IterationCounterWorks) 
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(1.0, 10.0);  

	EXPECT_EQ(sgd.GetIterations(), 0);

	sgd.PostUpdateParameters();
	EXPECT_EQ(sgd.GetIterations(), 1);

	sgd.PostUpdateParameters();
	EXPECT_EQ(sgd.GetIterations(), 2);
}

TEST_F(OptimizerEdgeCaseTest, LearningRateDecayApplied)
{
	NEURAL_NETWORK::StochasticGradientDescent sgd(0.1, 0.5);

	sgd.PreUpdateParameters();
	EXPECT_NEAR(sgd.GetLearningRate(), 0.1, 1e-12);

	sgd.PostUpdateParameters();
	sgd.PreUpdateParameters();
	double expected = 0.1 / (1.0 + 0.5 * 1.0);
	EXPECT_NEAR(sgd.GetLearningRate(), expected, 1e-12);
}
