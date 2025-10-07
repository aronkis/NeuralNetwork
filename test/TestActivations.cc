#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "ActivationReLU.h"
#include "ActivationSigmoid.h"
#include "ActivationSoftmax.h"
#include "ActivationLinear.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"
#include "LossCategoricalCrossentropy.h"
 
class ActivationTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		test_inputs = Eigen::MatrixXd(3, 4);
		test_inputs << -2.0, -1.0,  0.0,  1.0,
						2.0,  3.0, -0.5,  0.5,
						-1.5,  1.5,  2.5, -2.5;

		test_d_values = Eigen::MatrixXd(3, 4);
		test_d_values << 1.0,  0.5, -0.5, -1.0,
						-1.5,  2.0,  1.0,  0.0,
							0.5, -0.5,  1.5, -2.0;
	}

	Eigen::MatrixXd test_inputs;
	Eigen::MatrixXd test_d_values;
	const double tolerance = 1e-10;
};


class ReLUTest : public ActivationTest 
{};

TEST_F(ReLUTest, ForwardPassCorrectness) 
{
	NEURAL_NETWORK::ActivationReLU relu;
	relu.forward(test_inputs, false);
	const auto& output = relu.GetOutput();

	for (int i = 0; i < test_inputs.rows(); i++) 
	{
		for (int j = 0; j < test_inputs.cols(); j++) 
		{
			double expected = std::max(0.0, test_inputs(i, j));
			EXPECT_DOUBLE_EQ(output(i, j), expected);
		}
	}
}

TEST_F(ReLUTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::ActivationReLU relu;
	relu.forward(test_inputs, true);
	relu.backward(test_d_values);
	const auto& d_inputs = relu.GetDInput();

	for (int i = 0; i < test_inputs.rows(); i++) 
	{
		for (int j = 0; j < test_inputs.cols(); j++) 
		{
			double expected = (test_inputs(i, j) > 0) ? test_d_values(i, j) : 0.0;
			EXPECT_DOUBLE_EQ(d_inputs(i, j), expected);
		}
	}
}

TEST_F(ReLUTest, NegativeInputsProduceZero) 
{
	Eigen::MatrixXd negative_inputs = -Eigen::MatrixXd::Ones(2, 3);
	NEURAL_NETWORK::ActivationReLU relu;
	relu.forward(negative_inputs, false);
	const auto& output = relu.GetOutput();

	EXPECT_TRUE(output.isZero());
}

TEST_F(ReLUTest, PositiveInputsUnchanged) 
{
	Eigen::MatrixXd positive_inputs = Eigen::MatrixXd::Ones(2, 3);
	NEURAL_NETWORK::ActivationReLU relu;
	relu.forward(positive_inputs, false);
	const auto& output = relu.GetOutput();

	EXPECT_TRUE(output.isApprox(positive_inputs, tolerance));
}


class SigmoidTest : public ActivationTest 
{};

TEST_F(SigmoidTest, ForwardPassCorrectness) 
{
	NEURAL_NETWORK::ActivationSigmoid sigmoid;
	sigmoid.forward(test_inputs, false);
	const auto& output = sigmoid.GetOutput();

	for (int i = 0; i < test_inputs.rows(); i++) 
	{
		for (int j = 0; j < test_inputs.cols(); j++) 
		{
			double expected = 1.0 / (1.0 + std::exp(-test_inputs(i, j)));
			EXPECT_NEAR(output(i, j), expected, tolerance);
		}
	}
}

TEST_F(SigmoidTest, OutputRangeIs0To1) 
{
	NEURAL_NETWORK::ActivationSigmoid sigmoid;
	sigmoid.forward(test_inputs, false);
	const auto& output = sigmoid.GetOutput();

	for (int i = 0; i < output.rows(); i++) 
	{
		for (int j = 0; j < output.cols(); j++) 
		{
			EXPECT_GE(output(i, j), 0.0);
			EXPECT_LE(output(i, j), 1.0);
		}
	}
}

TEST_F(SigmoidTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::ActivationSigmoid sigmoid;
	sigmoid.forward(test_inputs, true);
	const auto& forward_output = sigmoid.GetOutput();

	sigmoid.backward(test_d_values);
	const auto& d_inputs = sigmoid.GetDInput();

	for (int i = 0; i < test_inputs.rows(); i++) 
	{
		for (int j = 0; j < test_inputs.cols(); j++) 
		{
			double sigmoid_out = forward_output(i, j);
			double expected = test_d_values(i, j) * sigmoid_out * (1.0 - sigmoid_out);
			EXPECT_NEAR(d_inputs(i, j), expected, tolerance);
		}
	}
}

TEST_F(SigmoidTest, ExtremeValuesHandled) 
{
	Eigen::MatrixXd extreme_inputs(2, 2);
	extreme_inputs << -100.0, 100.0,
					-50.0,  50.0;

					NEURAL_NETWORK::ActivationSigmoid sigmoid;
	sigmoid.forward(extreme_inputs, false);
	const auto& output = sigmoid.GetOutput();

	EXPECT_FALSE(output.hasNaN());
	EXPECT_TRUE(output.allFinite());

	EXPECT_NEAR(output(0, 0), 0.0, 1e-10);
	EXPECT_NEAR(output(0, 1), 1.0, 1e-10);
}


class SoftmaxTest : public ActivationTest 
{};

TEST_F(SoftmaxTest, ForwardPassSumsToOne) 
{
	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(test_inputs, false);
	const auto& output = softmax.GetOutput();

	for (int i = 0; i < output.rows(); i++) 
	{
		double row_sum = output.row(i).sum();
		EXPECT_NEAR(row_sum, 1.0, tolerance);
	}
}

TEST_F(SoftmaxTest, OutputIsPositive) 
{
	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(test_inputs, false);
	const auto& output = softmax.GetOutput();

	for (int i = 0; i < output.rows(); i++) 
	{
		for (int j = 0; j < output.cols(); j++) 
		{
			EXPECT_GT(output(i, j), 0.0);
		}
	}
}

TEST_F(SoftmaxTest, LargestInputProducesLargestOutput) 
{
	Eigen::MatrixXd inputs(1, 3);
	inputs << 1.0, 3.0, 2.0;  

	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(inputs, false);
	const auto& output = softmax.GetOutput();

	int max_input_idx, max_output_idx;
	inputs.row(0).maxCoeff(&max_input_idx);
	output.row(0).maxCoeff(&max_output_idx);

	EXPECT_EQ(max_input_idx, max_output_idx);
}

TEST_F(SoftmaxTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(test_inputs, true);
	softmax.backward(test_d_values);

	const auto& d_inputs = softmax.GetDInput();

	EXPECT_EQ(d_inputs.rows(), test_inputs.rows());
	EXPECT_EQ(d_inputs.cols(), test_inputs.cols());

	EXPECT_GT(d_inputs.cwiseAbs().sum(), 0.0);
}

TEST_F(SoftmaxTest, NumericalStability) 
{
	Eigen::MatrixXd large_inputs(1, 3);
	large_inputs << 1000.0, 1001.0, 999.0;

	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(large_inputs, false);
	const auto& output = softmax.GetOutput();

	EXPECT_FALSE(output.hasNaN());
	EXPECT_TRUE(output.allFinite());

	EXPECT_NEAR(output.row(0).sum(), 1.0, tolerance);
}


class LinearTest : public ActivationTest 
{};

TEST_F(LinearTest, ForwardPassIsIdentity) 
{
	NEURAL_NETWORK::ActivationLinear linear;
	linear.forward(test_inputs, false);
	const auto& output = linear.GetOutput();

	EXPECT_TRUE(output.isApprox(test_inputs, tolerance));
}

TEST_F(LinearTest, BackwardPassIsIdentity) 
{
	NEURAL_NETWORK::ActivationLinear linear;
	linear.forward(test_inputs, true);
	linear.backward(test_d_values);
	const auto& d_inputs = linear.GetDInput();

	EXPECT_TRUE(d_inputs.isApprox(test_d_values, tolerance));
}


class SoftmaxLossTest : public ActivationTest 
{};

TEST_F(SoftmaxLossTest, CombinedForwardAndLoss)
{

	Eigen::MatrixXi targets(3, 4);
	targets << 1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0;

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy softmax_loss;

	softmax_loss.storeTargets(targets);
	softmax_loss.forward(test_inputs, true);
	const auto& output = softmax_loss.GetOutput();

	for (int i = 0; i < output.rows(); i++) 
	{
		double row_sum = output.row(i).sum();
		EXPECT_NEAR(row_sum, 1.0, tolerance);
	}

	for (int i = 0; i < output.rows(); i++) 
	{
		for (int j = 0; j < output.cols(); j++) 
		{
			EXPECT_GT(output(i, j), 0.0);
		}
	}
}

TEST_F(SoftmaxLossTest, BackwardPassEfficiency)
{
	Eigen::MatrixXi targets(3, 4);
	targets << 1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0;

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy softmax_loss;

	softmax_loss.storeTargets(targets);
	softmax_loss.forward(test_inputs, true);
	softmax_loss.backward(test_d_values);

	const auto& d_inputs = softmax_loss.GetDInput();

	EXPECT_EQ(d_inputs.rows(), test_inputs.rows());
	EXPECT_EQ(d_inputs.cols(), test_inputs.cols());
}

TEST_F(SoftmaxLossTest, BackwardMatchesStandaloneOneHot)
{
	Eigen::MatrixXi targets(3, 1);
	targets << 0,
			1,
			2;

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy combined;
	combined.storeTargets(targets);
	combined.forward(test_inputs, true);

	Eigen::MatrixXd combined_output = combined.GetOutput();
	combined.backward(combined_output);
	Eigen::MatrixXd combined_grad = combined.GetDInput();

	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(test_inputs, true);
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;
	loss.CalculateLoss(softmax.GetOutput(), targets.cast<double>());
	loss.backward(softmax.GetOutput(), targets.cast<double>());
	softmax.backward(loss.GetDInput());

	EXPECT_TRUE(combined_grad.isApprox(softmax.GetDInput(), 1e-10));
}

TEST_F(SoftmaxLossTest, BackwardMatchesStandaloneOneHotMatrix)
{
	Eigen::MatrixXi targets(3, 4);
	targets << 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0;

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy combined;
	combined.storeTargets(targets);
	combined.forward(test_inputs, true);

	Eigen::MatrixXd combined_output = combined.GetOutput();
	combined.backward(combined_output);
	Eigen::MatrixXd combined_grad = combined.GetDInput();

	Eigen::MatrixXd targets_double = targets.cast<double>();

	NEURAL_NETWORK::ActivationSoftmax softmax;
	softmax.forward(test_inputs, true);
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;
	loss.CalculateLoss(softmax.GetOutput(), targets_double);
	loss.backward(softmax.GetOutput(), targets_double);
	softmax.backward(loss.GetDInput());

	EXPECT_TRUE(combined_grad.isApprox(softmax.GetDInput(), 1e-10));
}
