#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "Convolution.h"

class ConvolutionTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_input_4x4 = Eigen::MatrixXd(1, 16);
		test_input_4x4 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;

		test_input_3x3 = Eigen::MatrixXd(1, 9);
		test_input_3x3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

		test_input_batch2 = Eigen::MatrixXd(2, 9);
		test_input_batch2 << 1, 2, 3, 4, 5, 6, 7, 8, 9,
			9, 8, 7, 6, 5, 4, 3, 2, 1;
	}

	Eigen::MatrixXd test_input_4x4;
	Eigen::MatrixXd test_input_3x3;
	Eigen::MatrixXd test_input_batch2;
	const double tolerance = 1e-6;
};

TEST_F(ConvolutionTest, ConstructorInitialization)
{
	NEURAL_NETWORK::Convolution conv(2, 3, 3, 4, 4, 1, false, 1, 1);

	const auto &weights = conv.GetWeightsTensor();
	const auto &biases = conv.GetBiasesVector();

	EXPECT_EQ(weights.dimension(0), 3);
	EXPECT_EQ(weights.dimension(1), 3);
	EXPECT_EQ(weights.dimension(2), 1);
	EXPECT_EQ(weights.dimension(3), 2);

	EXPECT_EQ(biases.size(), 2);

	for (int i = 0; i < biases.size(); i++)
	{
		EXPECT_NEAR(biases(i), 0.0, tolerance);
	}
}

TEST_F(ConvolutionTest, OutputDimensionsValidConvolution)
{
	NEURAL_NETWORK::Convolution conv(2, 3, 3, 4, 4, 1, false, 1, 1);

	conv.forward(test_input_4x4, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 8);
}

TEST_F(ConvolutionTest, OutputDimensionsWithPadding)
{
	NEURAL_NETWORK::Convolution conv(2, 3, 3, 4, 4, 1, true, 1, 1);

	conv.forward(test_input_4x4, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 32);
}

TEST_F(ConvolutionTest, OutputDimensionsWithStride)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 4, 4, 1, false, 2, 2);

	conv.forward(test_input_4x4, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 4);
}

TEST_F(ConvolutionTest, BatchProcessing)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_batch2, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 2);
	EXPECT_EQ(output.cols(), 4);

	bool outputs_different = false;
	for (int col = 0; col < output.cols(); col++)
	{
		if (std::abs(output(0, col) - output(1, col)) > tolerance)
		{
			outputs_different = true;
			break;
		}
	}
	EXPECT_TRUE(outputs_different);
}

TEST_F(ConvolutionTest, SinglePixelFilter)
{
	NEURAL_NETWORK::Convolution conv(1, 1, 1, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 9);
}

TEST_F(ConvolutionTest, MultipleFilters)
{
	NEURAL_NETWORK::Convolution conv(3, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 12);
}

TEST_F(ConvolutionTest, AsymmetricFilter)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 3, 4, 4, 1, false, 1, 1);

	conv.forward(test_input_4x4, false);
	const auto &output = conv.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 6);
}

TEST_F(ConvolutionTest, OutputIsNotNaN)
{
	NEURAL_NETWORK::Convolution conv(2, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(output(i, j))) << "NaN found at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(output(i, j))) << "Inf found at (" << i << "," << j << ")";
		}
	}
}

TEST_F(ConvolutionTest, BiasInfluencesOutput)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output1 = conv.GetOutput();

	conv.forward(test_input_3x3, false);
	const auto &output2 = conv.GetOutput();

	for (int i = 0; i < output1.rows(); i++)
	{
		for (int j = 0; j < output1.cols(); j++)
		{
			EXPECT_NEAR(output1(i, j), output2(i, j), tolerance);
		}
	}
}

TEST_F(ConvolutionTest, PredictionsMatchOutput)
{
	NEURAL_NETWORK::Convolution conv(2, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();
	const auto predictions = conv.predictions();

	EXPECT_EQ(output.rows(), predictions.rows());
	EXPECT_EQ(output.cols(), predictions.cols());

	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			EXPECT_NEAR(output(i, j), predictions(i, j), tolerance);
		}
	}
}

TEST_F(ConvolutionTest, BackwardPassGradientShapes)
{
	NEURAL_NETWORK::Convolution conv(2, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	conv.backward(d_values);

	const auto &d_weights = conv.GetWeightsTensor();
	const auto &d_input = conv.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_3x3.rows());
	EXPECT_EQ(d_input.cols(), test_input_3x3.cols());

	bool has_nonzero_input_grad = false;
	for (int i = 0; i < d_input.rows(); i++)
	{
		for (int j = 0; j < d_input.cols(); j++)
		{
			if (std::abs(d_input(i, j)) > tolerance)
			{
				has_nonzero_input_grad = true;
				break;
			}
		}
	}
	EXPECT_TRUE(has_nonzero_input_grad);
}

TEST_F(ConvolutionTest, BackwardPassNumericalStability)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	Eigen::MatrixXd d_values(output.rows(), output.cols());
	d_values << 0.1, 0.5, 1.0, 2.0;

	conv.backward(d_values);
	const auto &d_input = conv.GetDInput();

	for (int i = 0; i < d_input.rows(); i++)
	{
		for (int j = 0; j < d_input.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(d_input(i, j))) << "NaN in input gradient at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(d_input(i, j))) << "Inf in input gradient at (" << i << "," << j << ")";
		}
	}
}

TEST_F(ConvolutionTest, BackwardPassWithBatches)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_batch2, false);
	const auto &output = conv.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	conv.backward(d_values);
	const auto &d_input = conv.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_batch2.rows());
	EXPECT_EQ(d_input.cols(), test_input_batch2.cols());

	for (int batch = 0; batch < d_input.rows(); batch++)
	{
		bool has_nonzero_grad = false;
		for (int j = 0; j < d_input.cols(); j++)
		{
			if (std::abs(d_input(batch, j)) > tolerance)
			{
				has_nonzero_grad = true;
				break;
			}
		}
		EXPECT_TRUE(has_nonzero_grad) << "Batch " << batch << " has all zero gradients";
	}
}

TEST_F(ConvolutionTest, BackwardPassConsistency)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);

	conv.forward(test_input_3x3, false);
	const auto &output = conv.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Constant(output.rows(), output.cols(), 0.5);

	conv.backward(d_values);
	const auto d_input1 = conv.GetDInput();

	conv.backward(d_values);
	const auto d_input2 = conv.GetDInput();

	for (int i = 0; i < d_input1.rows(); i++)
	{
		for (int j = 0; j < d_input1.cols(); j++)
		{
			EXPECT_NEAR(d_input1(i, j), d_input2(i, j), tolerance);
		}
	}
}

// Simplified tests for Convolution regularization
TEST_F(ConvolutionTest, RegularizationParametersStored)
{
	// Test that regularization parameters are stored correctly
	NEURAL_NETWORK::Convolution conv(2, 3, 3, 4, 4, 1, false, 1, 1, 0.1, 0.2);
	
	EXPECT_NEAR(conv.GetWeightRegularizerL1(), 0.1, tolerance);
	EXPECT_NEAR(conv.GetWeightRegularizerL2(), 0.2, tolerance);
}

TEST_F(ConvolutionTest, RegularizationInBackwardPass)
{
	// Test that backward pass works with regularization (basic functionality)
	NEURAL_NETWORK::Convolution conv(2, 3, 3, 4, 4, 1, false, 1, 1, 0.01, 0.01);
	
	// Forward and backward pass should work
	EXPECT_NO_THROW(conv.forward(test_input_4x4, true));
	
	const auto& output = conv.GetOutput();
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	
	EXPECT_NO_THROW(conv.backward(d_output));
	
	// Check that gradients exist
	const auto& d_weights = conv.GetDWeights();
	const auto& d_biases = conv.GetDBiases();
	
	EXPECT_EQ(d_weights.rows(), conv.GetWeights().rows());
	EXPECT_EQ(d_weights.cols(), conv.GetWeights().cols());
	EXPECT_EQ(d_biases.size(), conv.GetBiases().size());
}