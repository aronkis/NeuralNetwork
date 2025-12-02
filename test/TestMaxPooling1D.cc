#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "MaxPooling1D.h"

class MaxPooling1DTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_input_simple = Eigen::MatrixXd(1, 16);
		test_input_simple << 1, 0.5, 2, 1.0, 3, 1.5, 4, 2.0, 5, 2.5, 6, 3.0, 7, 3.5, 8, 4.0;

		test_input_batch = Eigen::MatrixXd(2, 8);
		test_input_batch << 1, 2, 3, 4, 5, 6, 7, 8,
						   8, 7, 6, 5, 4, 3, 2, 1;

		test_input_multichannel = Eigen::MatrixXd(1, 18);
		test_input_multichannel << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18;

		test_input_known = Eigen::MatrixXd(1, 6);
		test_input_known << 1, 3, 2, 6, 4, 5;
	}

	Eigen::MatrixXd test_input_simple;
	Eigen::MatrixXd test_input_batch;
	Eigen::MatrixXd test_input_multichannel;
	Eigen::MatrixXd test_input_known;
	const double tolerance = 1e-6;
};

TEST_F(MaxPooling1DTest, ConstructorInitialization)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	EXPECT_EQ(maxpool.GetPoolSize(), 2);
	EXPECT_EQ(maxpool.GetStride(), 2);
	EXPECT_EQ(maxpool.GetInputLength(), 8);
	EXPECT_EQ(maxpool.GetInputChannels(), 2);
	EXPECT_EQ(maxpool.GetOutputLength(), 4); 
}

TEST_F(MaxPooling1DTest, ForwardPassOutputDimensions)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 1);          
	EXPECT_EQ(output.cols(), 4 * 2);      
}

TEST_F(MaxPooling1DTest, ForwardPassDifferentPoolSizes)
{
	NEURAL_NETWORK::MaxPooling1D maxpool2(1, 2, 8, 1, 1);
	maxpool2.forward(test_input_batch.row(0).transpose().transpose(), false);
	auto output2 = maxpool2.GetOutput();
	EXPECT_EQ(output2.cols(), 7 * 1); 

	NEURAL_NETWORK::MaxPooling1D maxpool4(1, 4, 8, 1, 1);
	maxpool4.forward(test_input_batch.row(0).transpose().transpose(), false);
	auto output4 = maxpool4.GetOutput();
	EXPECT_EQ(output4.cols(), 5 * 1); 
}

TEST_F(MaxPooling1DTest, ForwardPassDifferentStrides)
{
	NEURAL_NETWORK::MaxPooling1D maxpool_s1(1, 2, 8, 1, 1);
	maxpool_s1.forward(test_input_batch.row(0).transpose().transpose(), false);
	auto output_s1 = maxpool_s1.GetOutput();
	EXPECT_EQ(output_s1.cols(), 7 * 1); 

	NEURAL_NETWORK::MaxPooling1D maxpool_s3(1, 2, 8, 1, 3);
	maxpool_s3.forward(test_input_batch.row(0).transpose().transpose(), false);
	auto output_s3 = maxpool_s3.GetOutput();
	EXPECT_EQ(output_s3.cols(), 3 * 1); 
}

TEST_F(MaxPooling1DTest, BatchProcessing)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(2, 2, 8, 1, 2);

	maxpool.forward(test_input_batch, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 2);          
	EXPECT_EQ(output.cols(), 4 * 1);      

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

TEST_F(MaxPooling1DTest, MultiChannelInput)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 6, 3, 2);

	maxpool.forward(test_input_multichannel, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 1);          
	EXPECT_EQ(output.cols(), 3 * 3);      
}

TEST_F(MaxPooling1DTest, OutputNumericalStability)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto& output = maxpool.GetOutput();

	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(output(i, j))) << "NaN found at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(output(i, j))) << "Inf found at (" << i << "," << j << ")";
		}
	}
}

TEST_F(MaxPooling1DTest, PredictionsMatchOutput)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto& output = maxpool.GetOutput();
	const auto predictions = maxpool.predictions();

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

TEST_F(MaxPooling1DTest, MathematicalCorrectness)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 6, 1, 2);

	maxpool.forward(test_input_known, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 3); 

	EXPECT_NEAR(output(0, 0), 3.0, tolerance); 
	EXPECT_NEAR(output(0, 1), 6.0, tolerance); 
	EXPECT_NEAR(output(0, 2), 5.0, tolerance); 
}

TEST_F(MaxPooling1DTest, BackwardPassGradientShapes)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto& output = maxpool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	maxpool.backward(d_values);

	const auto& d_input = maxpool.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_simple.rows());
	EXPECT_EQ(d_input.cols(), test_input_simple.cols());
}

TEST_F(MaxPooling1DTest, BackwardPassGradientFlow)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 6, 1, 2);

	maxpool.forward(test_input_known, false);
	const auto& output = maxpool.GetOutput();

	
	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	maxpool.backward(d_values);

	const auto& d_input = maxpool.GetDInput();

	EXPECT_NEAR(d_input(0, 0), 0.0, tolerance); 
	EXPECT_NEAR(d_input(0, 1), 1.0, tolerance); 
	EXPECT_NEAR(d_input(0, 2), 0.0, tolerance); 
	EXPECT_NEAR(d_input(0, 3), 1.0, tolerance); 
	EXPECT_NEAR(d_input(0, 4), 0.0, tolerance); 
	EXPECT_NEAR(d_input(0, 5), 1.0, tolerance); 
}

TEST_F(MaxPooling1DTest, BackwardPassNumericalStability)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto& output = maxpool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Random(output.rows(), output.cols());

	maxpool.backward(d_values);
	const auto& d_input = maxpool.GetDInput();

	for (int i = 0; i < d_input.rows(); i++)
	{
		for (int j = 0; j < d_input.cols(); j++)
		{
			EXPECT_FALSE(std::isnan(d_input(i, j))) << "NaN in input gradient at (" << i << "," << j << ")";
			EXPECT_FALSE(std::isinf(d_input(i, j))) << "Inf in input gradient at (" << i << "," << j << ")";
		}
	}
}

TEST_F(MaxPooling1DTest, BackwardPassWithBatches)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(2, 2, 8, 1, 2);

	maxpool.forward(test_input_batch, false);
	const auto& output = maxpool.GetOutput();

	Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	maxpool.backward(d_values);
	const auto& d_input = maxpool.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input_batch.rows());
	EXPECT_EQ(d_input.cols(), test_input_batch.cols());

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

TEST_F(MaxPooling1DTest, DynamicBatchSize)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 1, 2); 

	Eigen::MatrixXd input1(1, 8);
	input1 << 1, 2, 3, 4, 5, 6, 7, 8;

	maxpool.forward(input1, false);
	auto output1 = maxpool.GetOutput();
	EXPECT_EQ(output1.rows(), 1);

	Eigen::MatrixXd input3(3, 8);
	input3 << 1, 2, 3, 4, 5, 6, 7, 8,
			  8, 7, 6, 5, 4, 3, 2, 1,
			  2, 4, 6, 8, 1, 3, 5, 7;

	maxpool.forward(input3, false);
	auto output3 = maxpool.GetOutput();
	EXPECT_EQ(output3.rows(), 3);
}

TEST_F(MaxPooling1DTest, ForwardPassConsistency)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);

	maxpool.forward(test_input_simple, false);
	const auto output1 = maxpool.GetOutput();

	maxpool.forward(test_input_simple, false);
	const auto output2 = maxpool.GetOutput();

	EXPECT_TRUE(output1.isApprox(output2, tolerance));
}

TEST_F(MaxPooling1DTest, PoolSizeEqualsInputLength)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 6, 6, 1, 1);

	maxpool.forward(test_input_known, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 1); 

	EXPECT_NEAR(output(0, 0), 6.0, tolerance); 
}

TEST_F(MaxPooling1DTest, OverlappingPools)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 3, 6, 1, 1); 

	maxpool.forward(test_input_known, false);
	const auto& output = maxpool.GetOutput();

	EXPECT_EQ(output.rows(), 1);
	EXPECT_EQ(output.cols(), 4); 

	EXPECT_NEAR(output(0, 0), 3.0, tolerance); 
	EXPECT_NEAR(output(0, 1), 6.0, tolerance); 
	EXPECT_NEAR(output(0, 2), 6.0, tolerance); 
	EXPECT_NEAR(output(0, 3), 6.0, tolerance); 
}

TEST_F(MaxPooling1DTest, LayerBaseCompliance)
{
	NEURAL_NETWORK::MaxPooling1D maxpool(1, 2, 8, 2, 2);
	
	EXPECT_EQ(maxpool.GetWeightRegularizerL1(), 0.0);
	EXPECT_EQ(maxpool.GetWeightRegularizerL2(), 0.0);
	EXPECT_EQ(maxpool.GetBiasRegularizerL1(), 0.0);
	EXPECT_EQ(maxpool.GetBiasRegularizerL2(), 0.0);

	
	EXPECT_EQ(maxpool.GetWeights().size(), 0);
	EXPECT_EQ(maxpool.GetBiases().size(), 0);
	
	Eigen::MatrixXd dummy_weights = Eigen::MatrixXd::Zero(1, 1);
	Eigen::RowVectorXd dummy_biases = Eigen::RowVectorXd::Zero(1);
	EXPECT_NO_THROW(maxpool.SetParameters(dummy_weights, dummy_biases));
}