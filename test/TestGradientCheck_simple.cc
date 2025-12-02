#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "Convolution.h"
#include "MaxPooling.h"
#include "LayerDense.h"
#include "ActivationReLU.h"

class GradientCheckTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		tolerance = 1e-3;
		test_input_2x2 = Eigen::MatrixXd(1, 4);
		test_input_2x2 << 0.1, 0.2, 0.3, 0.4;
	}
	
	Eigen::MatrixXd test_input_2x2;
	double tolerance;
};

TEST_F(GradientCheckTest, ConvolutionBasicGradients)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1);
	
	conv.forward(test_input_2x2, true);
	const auto& output = conv.GetOutput();
	
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	conv.backward(d_output);
	
	const auto& d_weights = conv.GetDWeights();
	const auto& d_biases = conv.GetDBiases();
	const auto& weights = conv.GetWeights();
	const auto& biases = conv.GetBiases();
	
	EXPECT_EQ(d_weights.rows(), weights.rows());
	EXPECT_EQ(d_weights.cols(), weights.cols());
	EXPECT_EQ(d_biases.size(), biases.size());
	
	EXPECT_GT(d_weights.norm(), 0.0);
	EXPECT_GT(d_biases.norm(), 0.0);
}

TEST_F(GradientCheckTest, MaxPoolingGradients)
{
	NEURAL_NETWORK::MaxPooling pool(2, 2, 1, 1, 2, 2);
	
	pool.forward(test_input_2x2, true);
	const auto& output = pool.GetOutput();
	
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	pool.backward(d_output);
	
	const auto& d_inputs = pool.GetDInputs();
	EXPECT_EQ(d_inputs.rows(), test_input_2x2.rows());
	EXPECT_EQ(d_inputs.cols(), test_input_2x2.cols());
}

TEST_F(GradientCheckTest, DenseLayerGradients)
{
	NEURAL_NETWORK::LayerDense dense(4, 3);
	
	dense.forward(test_input_2x2);
	const auto& output = dense.GetOutput();
	
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	dense.backward(d_output);
	
	const auto& d_weights = dense.GetDWeights();
	const auto& d_biases = dense.GetDBiases();
	const auto& d_inputs = dense.GetDInputs();
	const auto& weights = dense.GetWeights();
	const auto& biases = dense.GetBiases();
	
	EXPECT_EQ(d_weights.rows(), weights.rows());
	EXPECT_EQ(d_weights.cols(), weights.cols());
	EXPECT_EQ(d_biases.size(), biases.size());
	EXPECT_EQ(d_inputs.rows(), test_input_2x2.rows());
	EXPECT_EQ(d_inputs.cols(), test_input_2x2.cols());
}

TEST_F(GradientCheckTest, ReLUActivationGradients)
{
	NEURAL_NETWORK::ActivationReLU relu;
	
	Eigen::MatrixXd mixed_input(1, 4);
	mixed_input << -1.0, 0.5, -0.2, 1.5;
	
	relu.forward(mixed_input);
	const auto& output = relu.GetOutput();
	
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	relu.backward(d_output);
	
	const auto& d_inputs = relu.GetDInputs();
	EXPECT_EQ(d_inputs.rows(), mixed_input.rows());
	EXPECT_EQ(d_inputs.cols(), mixed_input.cols());
	
	EXPECT_NEAR(d_inputs(0, 0), 0.0, tolerance); 
	EXPECT_NEAR(d_inputs(0, 1), 1.0, tolerance); 
	EXPECT_NEAR(d_inputs(0, 2), 0.0, tolerance); 
	EXPECT_NEAR(d_inputs(0, 3), 1.0, tolerance); 
}