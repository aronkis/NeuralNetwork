#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "MaxPooling.h"

class MaxPoolingTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_input_4x4 = Eigen::MatrixXd(1, 16);
		test_input_4x4 <<  1,  2,  3,  4, 
                           5,  6,  7,  8, 
                           9, 10, 11, 12, 
                          13, 14, 15, 16;

		test_input_6x6 = Eigen::MatrixXd(1, 36);
		test_input_6x6 <<  1,  2,  3,  4,  5,  6, 
                           7,  8,  9, 10, 11, 12, 
                          13, 14, 15, 16, 17, 18, 
                          19, 20, 21, 22, 23, 24, 
                          25, 26, 27, 28, 29, 30, 
                          31, 32, 33, 34, 35, 36;

		test_input_batch2 = Eigen::MatrixXd(2, 9);
		test_input_batch2 << 1, 2, 3, 4, 5, 6, 7, 8, 9,
							 9, 8, 7, 6, 5, 4, 3, 2, 1;

		test_input_2channels = Eigen::MatrixXd(1, 32);
		test_input_2channels <<  1,  2,  3,  4,  5,  6,  7,  8, 
                                 9, 10, 11, 12, 13, 14, 15, 16, 
                                16, 15, 14, 13, 12, 11, 10,  9, 
                                 8,  7,  6,  5,  4,  3,  2,  1;
	}

	Eigen::MatrixXd test_input_4x4;
	Eigen::MatrixXd test_input_6x6;
	Eigen::MatrixXd test_input_batch2;
	Eigen::MatrixXd test_input_2channels;
	const double tolerance = 1e-6;
};

TEST_F(MaxPoolingTest, ConstructorInitialization)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	EXPECT_EQ(pool.GetInputHeight(), 4);
	EXPECT_EQ(pool.GetInputWidth(), 4);
	EXPECT_EQ(pool.GetInputChannels(), 1);
	EXPECT_EQ(pool.GetStride(), 2);
}

TEST_F(MaxPoolingTest, OutputDimensions2x2Pool)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 4);
	EXPECT_EQ(pool.GetOutputHeight(), 2);
	EXPECT_EQ(pool.GetOutputWidth(), 2);
}

TEST_F(MaxPoolingTest, OutputDimensions3x3Pool)
{
	NEURAL_NETWORK::MaxPooling pool(1, 3, 6, 6, 1, 3);

	pool.forward(test_input_6x6, false);
	const auto& output = pool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 4);
	EXPECT_EQ(pool.GetOutputHeight(), 2);
	EXPECT_EQ(pool.GetOutputWidth(), 2);
}

TEST_F(MaxPoolingTest, OutputDimensionsWithStride1)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 1);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 9);
	EXPECT_EQ(pool.GetOutputHeight(), 3);
	EXPECT_EQ(pool.GetOutputWidth(), 3);
}

TEST_F(MaxPoolingTest, MaxPoolingCorrectValues2x2)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	// Expected max values:
	// Top-left 2x2 region (1,2,5,6) -> max = 6
	// Top-right 2x2 region (3,4,7,8) -> max = 8
	// Bottom-left 2x2 region (9,10,13,14) -> max = 14
	// Bottom-right 2x2 region (11,12,15,16) -> max = 16
	
	EXPECT_NEAR(output(0, 0), 6.0, tolerance);
	EXPECT_NEAR(output(0, 1), 8.0, tolerance);
	EXPECT_NEAR(output(0, 2), 14.0, tolerance);
	EXPECT_NEAR(output(0, 3), 16.0, tolerance);
}

TEST_F(MaxPoolingTest, MaxPoolingCorrectValues3x3)
{
	NEURAL_NETWORK::MaxPooling pool(1, 3, 6, 6, 1, 3);

	pool.forward(test_input_6x6, false);
	const auto& output = pool.GetOutput();

	// Top-left 3x3 region (1-9, 7-15, 13-21) -> max = 15
	// Top-right 3x3 region (4-12, 10-18, 16-24) -> max = 18
	// Bottom-left 3x3 region (19-27, 25-33, 31-39) -> max = 33
	// Bottom-right 3x3 region (22-30, 28-36, 34-42) -> max = 36
	
	EXPECT_NEAR(output(0, 0), 15.0, tolerance);
	EXPECT_NEAR(output(0, 1), 18.0, tolerance);
	EXPECT_NEAR(output(0, 2), 33.0, tolerance);
	EXPECT_NEAR(output(0, 3), 36.0, tolerance);
}

TEST_F(MaxPoolingTest, BatchProcessing)
{
	NEURAL_NETWORK::MaxPooling pool(2, 2, 3, 3, 1, 1);

	pool.forward(test_input_batch2, false);
	const auto& output = pool.GetOutput();

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

TEST_F(MaxPoolingTest, MultipleChannels)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 2, 2);

	pool.forward(test_input_2channels, false);
	const auto& output = pool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 8);
	EXPECT_EQ(pool.GetOutputHeight(), 2);
	EXPECT_EQ(pool.GetOutputWidth(), 2);
}

TEST_F(MaxPoolingTest, OutputIsNotNaN)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(output(i, j))) << "NaN found at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(output(i, j))) << "Inf found at (" << i << "," << j << ")";
		}
	}
}

TEST_F(MaxPoolingTest, OutputMatchesPredictions)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();
	const auto predictions = pool.predictions();

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

TEST_F(MaxPoolingTest, MaxPoolingNonOverlapping)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	// Input as 4x4 image (row-major order):
	// [1, 0, 0, 2]
	// [0, 0, 0, 0]
	// [0, 0, 3, 0]
	// [0, 0, 0, 4]
	Eigen::MatrixXd input(1, 16);
	input << 1, 0, 0, 2, 
             0, 0, 0, 0, 
             0, 0, 3, 0, 
             0, 0, 0, 4;

	pool.forward(input, false);
	const auto& output = pool.GetOutput();

	EXPECT_NEAR(output(0, 0), 1.0, tolerance);
	EXPECT_NEAR(output(0, 1), 2.0, tolerance);
	EXPECT_NEAR(output(0, 2), 0.0, tolerance);
	EXPECT_NEAR(output(0, 3), 4.0, tolerance);
}

TEST_F(MaxPoolingTest, BackwardPassGradientShapes)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	pool.backward(d_values);

	const auto& d_input = pool.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_4x4.rows());
	EXPECT_EQ(d_input.cols(), test_input_4x4.cols());
}

TEST_F(MaxPoolingTest, BackwardPassRoutesToMaxIndices)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	Eigen::MatrixXd input(1, 16);
	input << 1,  0, 0,  1,
             0, 10, 0, 20, 
             0,  0, 1,  0, 
             0, 30, 0, 40;

	pool.forward(input, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	pool.backward(d_values);
	const auto& d_input = pool.GetDInput();

	EXPECT_NEAR(d_input(0, 5), 1.0, tolerance);
	EXPECT_NEAR(d_input(0, 7), 1.0, tolerance);
	EXPECT_NEAR(d_input(0, 13), 1.0, tolerance);
	EXPECT_NEAR(d_input(0, 15), 1.0, tolerance);

	EXPECT_NEAR(d_input(0, 0), 0.0, tolerance);
	EXPECT_NEAR(d_input(0, 1), 0.0, tolerance);
	EXPECT_NEAR(d_input(0, 2), 0.0, tolerance);
}

TEST_F(MaxPoolingTest, BackwardPassNumericalStability)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values(output.rows(), output.cols());
	d_values << 0.1, 0.5, 1.0, 2.0;

	pool.backward(d_values);
	const auto& d_input = pool.GetDInput();

	for (int i = 0; i < d_input.rows(); i++)
	{
		for (int j = 0; j < d_input.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(d_input(i, j))) << "NaN in input gradient at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(d_input(i, j))) << "Inf in input gradient at (" << i << "," << j << ")";
		}
	}
}

TEST_F(MaxPoolingTest, BackwardPassWithBatches)
{
	NEURAL_NETWORK::MaxPooling pool(2, 2, 3, 3, 1, 1);

	pool.forward(test_input_batch2, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	pool.backward(d_values);
	const auto& d_input = pool.GetDInput();

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

TEST_F(MaxPoolingTest, BackwardPassConsistency)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Constant(output.rows(), output.cols(), 0.5);

	pool.backward(d_values);
	const auto d_input1 = pool.GetDInput();

	pool.backward(d_values);
	const auto d_input2 = pool.GetDInput();

	for (int i = 0; i < d_input1.rows(); i++)
	{
		for (int j = 0; j < d_input1.cols(); j++)
		{
			EXPECT_NEAR(d_input1(i, j), d_input2(i, j), tolerance);
		}
	}
}

TEST_F(MaxPoolingTest, BackwardPassWithMultipleChannels)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 2, 2);

	pool.forward(test_input_2channels, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	pool.backward(d_values);
	const auto& d_input = pool.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_2channels.rows());
	EXPECT_EQ(d_input.cols(), test_input_2channels.cols());

	bool has_nonzero_grad = false;
	for (int j = 0; j < d_input.cols(); j++)
	{
		if (std::abs(d_input(0, j)) > tolerance)
		{
			has_nonzero_grad = true;
			break;
		}
	}
	EXPECT_TRUE(has_nonzero_grad);
}

TEST_F(MaxPoolingTest, BackwardGradientPropagation)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	Eigen::MatrixXd d_values(1, 4);
	d_values << 1.0, 2.0, 3.0, 4.0;

	pool.backward(d_values);
	const auto& d_input = pool.GetDInput();

	double input_grad_sum = d_input.sum();
	double output_grad_sum = d_values.sum();

	EXPECT_NEAR(input_grad_sum, output_grad_sum, tolerance);
}

TEST_F(MaxPoolingTest, OverlappingPoolRegions)
{
	NEURAL_NETWORK::MaxPooling pool(1, 3, 4, 4, 1, 1);

	pool.forward(test_input_4x4, false);
	const auto& output = pool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 4);
	EXPECT_EQ(pool.GetOutputHeight(), 2);
	EXPECT_EQ(pool.GetOutputWidth(), 2);
}

TEST_F(MaxPoolingTest, ZeroInputHandling)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	Eigen::MatrixXd zero_input = Eigen::MatrixXd::Zero(1, 16);

	pool.forward(zero_input, false);
	const auto& output = pool.GetOutput();

	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			EXPECT_NEAR(output(i, j), 0.0, tolerance);
		}
	}
}

TEST_F(MaxPoolingTest, NegativeInputHandling)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 4, 4, 1, 2);

	Eigen::MatrixXd negative_input(1, 16);
	negative_input << -16, -15, -14, -13, 
                      -12, -11, -10,  -9, 
                       -8,  -7,  -6,  -5, 
                       -4,  -3,  -2,  -1;

	pool.forward(negative_input, false);
	const auto& output = pool.GetOutput();

	EXPECT_NEAR(output(0, 0), -11.0, tolerance);
	EXPECT_NEAR(output(0, 1), -9.0, tolerance);
	EXPECT_NEAR(output(0, 2), -3.0, tolerance);
	EXPECT_NEAR(output(0, 3), -1.0, tolerance);
}
