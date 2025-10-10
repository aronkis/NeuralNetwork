
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "LayerDense.h"

class LayerDenseTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		n_inputs = 3;
		n_neurons = 4;
		layer = std::make_unique<NEURAL_NETWORK::LayerDense>(n_inputs, n_neurons);

		test_inputs = Eigen::MatrixXd::Random(2, n_inputs);
		test_d_values = Eigen::MatrixXd::Random(2, n_neurons);
	}

	int n_inputs;
	int n_neurons;
	std::unique_ptr<NEURAL_NETWORK::LayerDense> layer;
	Eigen::MatrixXd test_inputs;
	Eigen::MatrixXd test_d_values;
};

TEST_F(LayerDenseTest, ConstructorInitializesCorrectDimensions) 
{
	const auto& weights = layer->GetWeights();
	const auto& biases = layer->GetBiases();

	EXPECT_EQ(weights.rows(), n_inputs);
	EXPECT_EQ(weights.cols(), n_neurons);
	EXPECT_EQ(biases.cols(), n_neurons);
	EXPECT_EQ(biases.rows(), 1);
}

TEST_F(LayerDenseTest, WeightInitializationIsReasonable) 
{
	const auto& weights = layer->GetWeights();

	EXPECT_LT(weights.cwiseAbs().maxCoeff(), 5.0);  
	EXPECT_GT(weights.cwiseAbs().maxCoeff(), 0.0);

	const auto& biases = layer->GetBiases();
	EXPECT_DOUBLE_EQ(biases.sum(), 0.0);
}

TEST_F(LayerDenseTest, ForwardPassProducesCorrectOutputDimensions) 
{
	layer->forward(test_inputs, false);
	const auto& output = layer->GetOutput();

	EXPECT_EQ(output.rows(), test_inputs.rows());  
	EXPECT_EQ(output.cols(), n_neurons);           
}

TEST_F(LayerDenseTest, ForwardPassMathematicalCorrectness) 
{
	Eigen::MatrixXd inputs(2, 3);
	inputs << 1.0, 2.0, 3.0,
				4.0, 5.0, 6.0;

	Eigen::MatrixXd weights(3, 2);
	weights << 0.1, 0.2,
				0.3, 0.4,
				0.5, 0.6;

	Eigen::RowVectorXd biases(2);
	biases << 0.1, 0.2;

	layer = std::make_unique<NEURAL_NETWORK::LayerDense>(3, 2);
	layer->SetParameters(weights, biases);

	layer->forward(inputs, false);
	const auto& output = layer->GetOutput();

	Eigen::MatrixXd expected = inputs * weights + biases.replicate(inputs.rows(), 1);

	EXPECT_TRUE(output.isApprox(expected, 1e-10));
}

TEST_F(LayerDenseTest, BackwardPassComputesGradients) 
{
	layer->forward(test_inputs, true);

	layer->backward(test_d_values);

	const auto& d_weights = layer->GetDWeights();
	const auto& d_biases = layer->GetDBiases();
	const auto& d_inputs = layer->GetDInput();

	EXPECT_EQ(d_weights.rows(), n_inputs);
	EXPECT_EQ(d_weights.cols(), n_neurons);
	EXPECT_EQ(d_biases.cols(), n_neurons);
	EXPECT_EQ(d_inputs.rows(), test_inputs.rows());
	EXPECT_EQ(d_inputs.cols(), n_inputs);
}

TEST_F(LayerDenseTest, BackwardPassGradientMagnitudes) 
{
	layer->forward(test_inputs, true);
	layer->backward(test_d_values);

	const auto& d_weights = layer->GetDWeights();
	const auto& d_biases = layer->GetDBiases();

	EXPECT_GT(d_weights.cwiseAbs().sum(), 0.0);
	EXPECT_GT(d_biases.cwiseAbs().sum(), 0.0);
}

TEST_F(LayerDenseTest, RegularizationParametersAreStoredCorrectly) 
{
	double w_l1 = 0.01, w_l2 = 0.02, b_l1 = 0.03, b_l2 = 0.04;
	auto reg_layer = std::make_unique<NEURAL_NETWORK::LayerDense>(n_inputs, n_neurons, w_l1, w_l2, b_l1, b_l2);

	EXPECT_DOUBLE_EQ(reg_layer->GetWeightRegularizerL1(), w_l1);
	EXPECT_DOUBLE_EQ(reg_layer->GetWeightRegularizerL2(), w_l2);
	EXPECT_DOUBLE_EQ(reg_layer->GetBiasRegularizerL1(), b_l1);
	EXPECT_DOUBLE_EQ(reg_layer->GetBiasRegularizerL2(), b_l2);
}

TEST_F(LayerDenseTest, ParameterGettersReturnCorrectValues)
{
	auto [weights_copy, biases_copy] = layer->GetParameters();
	const auto& weights_direct = layer->GetWeights();
	const auto& biases_direct = layer->GetBiases();

	EXPECT_TRUE(weights_copy.isApprox(weights_direct));
	EXPECT_TRUE(biases_copy.isApprox(biases_direct));
}

TEST_F(LayerDenseTest, SetParametersChangesWeightsAndBiases) 
{
	Eigen::MatrixXd new_weights = Eigen::MatrixXd::Ones(n_inputs, n_neurons);
	Eigen::RowVectorXd new_biases = Eigen::RowVectorXd::Constant(n_neurons, 0.5);

	layer->SetParameters(new_weights, new_biases);

	const auto& stored_weights = layer->GetWeights();
	const auto& stored_biases = layer->GetBiases();

	EXPECT_TRUE(stored_weights.isApprox(new_weights));
	EXPECT_TRUE(stored_biases.isApprox(new_biases));
}

TEST_F(LayerDenseTest, MomentumAndCacheInitialization) 
{
	const auto& weight_momentums = layer->GetWeightMomentums();
	const auto& bias_momentums = layer->GetBiasMomentums();
	const auto& weight_caches = layer->GetWeightCaches();
	const auto& bias_caches = layer->GetBiasCaches();

	EXPECT_TRUE(weight_momentums.isZero());
	EXPECT_TRUE(bias_momentums.isZero());
	EXPECT_TRUE(weight_caches.isZero());
	EXPECT_TRUE(bias_caches.isZero());
}

TEST_F(LayerDenseTest, PredictionsMethodReturnsOutput) 
{
	layer->forward(test_inputs, false);
	const auto& output = layer->GetOutput();
	const auto& predictions = layer->predictions();

	EXPECT_TRUE(output.isApprox(predictions));
}

TEST_F(LayerDenseTest, TrainingModeAffectsInputStorage) 
{
	Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(2, n_inputs);

	layer->forward(inputs, true);
	layer->backward(test_d_values);  

	const auto& d_inputs = layer->GetDInput();
	EXPECT_EQ(d_inputs.rows(), inputs.rows());
	EXPECT_EQ(d_inputs.cols(), inputs.cols());
}

TEST_F(LayerDenseTest, SingleNeuronLayer) 
{
	auto single_layer = std::make_unique<NEURAL_NETWORK::LayerDense>(n_inputs, 1);
	single_layer->forward(test_inputs, false);

	const auto& output = single_layer->GetOutput();
	EXPECT_EQ(output.cols(), 1);
	EXPECT_EQ(output.rows(), test_inputs.rows());
}

TEST_F(LayerDenseTest, SingleInputLayer) 
{
	Eigen::MatrixXd single_input = Eigen::MatrixXd::Random(2, 1);
	auto single_input_layer = std::make_unique<NEURAL_NETWORK::LayerDense>(1, n_neurons);

	single_input_layer->forward(single_input, false);
	const auto& output = single_input_layer->GetOutput();

	EXPECT_EQ(output.cols(), n_neurons);
	EXPECT_EQ(output.rows(), single_input.rows());
}
