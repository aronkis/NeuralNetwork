#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "Convolution1D.h"

class Convolution1DGradientTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_input = Eigen::MatrixXd(2, 3);
		test_input << 1.0, 2.0, 3.0,
					  0.5, 1.5, 2.5;

		epsilon = 1e-5;
		tolerance = 1e-3; 
	}

	
	double computeNumericalGradient(NEURAL_NETWORK::Convolution1D& conv,
								   const Eigen::MatrixXd& input,
								   const Eigen::MatrixXd& target,
								   int param_idx, bool is_weight = true)
	{
		auto [weights, biases] = conv.GetParameters();

		conv.forward(input, true);
		auto output = conv.GetOutput();
		double loss_original = 0.5 * (output - target).array().square().sum();

		if (is_weight)
		{
			weights.data()[param_idx] += epsilon;
			conv.SetParameters(weights, biases);
		}
		else
		{
			biases.data()[param_idx] += epsilon;
			conv.SetParameters(weights, biases);
		}

		conv.forward(input, true);
		auto output_perturbed = conv.GetOutput();
		double loss_perturbed = 0.5 * (output_perturbed - target).array().square().sum();

		if (is_weight)
		{
			weights.data()[param_idx] -= epsilon;
		}
		else
		{
			biases.data()[param_idx] -= epsilon;
		}
		conv.SetParameters(weights, biases);

		return (loss_perturbed - loss_original) / epsilon;
	}

	Eigen::MatrixXd test_input;
	double epsilon;
	double tolerance;
};

TEST_F(Convolution1DGradientTest, WeightGradientCheck)
{
	NEURAL_NETWORK::Convolution1D conv(2, 2, 3, 1, 0, 1); 

	conv.forward(test_input, true);
	auto output_size = conv.GetOutput();
	Eigen::MatrixXd target = Eigen::MatrixXd::Ones(output_size.rows(), output_size.cols());

	conv.forward(test_input, true);
	auto output = conv.GetOutput();
	Eigen::MatrixXd d_output = output - target; 
	conv.backward(d_output);

	const auto& analytical_grads = conv.GetDWeights();

	int num_checks = std::min(4, static_cast<int>(analytical_grads.size()));
	for (int i = 0; i < num_checks; i++)
	{
		double numerical_grad = computeNumericalGradient(conv, test_input, target, i, true);
		double analytical_grad = analytical_grads.data()[i];

		EXPECT_NEAR(analytical_grad, numerical_grad, tolerance)
			<< "Weight gradient mismatch at index " << i
			<< " (analytical: " << analytical_grad
			<< ", numerical: " << numerical_grad << ")";
	}
}

TEST_F(Convolution1DGradientTest, BiasGradientCheck)
{
	NEURAL_NETWORK::Convolution1D conv(2, 2, 3, 1, 0, 1); 

	std::mt19937 gen(42);
	std::normal_distribution<double> dist(0.0, 1.0);

	conv.forward(test_input, true);
	auto output_size = conv.GetOutput();
	Eigen::MatrixXd target = Eigen::MatrixXd::Zero(output_size.rows(), output_size.cols());
	for (int i = 0; i < target.rows(); i++)
	{
		for (int j = 0; j < target.cols(); j++)
		{
			target(i, j) = dist(gen);
		}
	}

	conv.forward(test_input, true);
	auto output = conv.GetOutput();
	Eigen::MatrixXd d_output = output - target; 
	conv.backward(d_output);

	const auto& analytical_grads = conv.GetDBiases();

	for (int i = 0; i < analytical_grads.size(); i++)
	{
		double numerical_grad = computeNumericalGradient(conv, test_input, target, i, false);
		double analytical_grad = analytical_grads.data()[i];

		EXPECT_NEAR(analytical_grad, numerical_grad, tolerance)
			<< "Bias gradient mismatch at index " << i
			<< " (analytical: " << analytical_grad
			<< ", numerical: " << numerical_grad << ")";
	}
}

TEST_F(Convolution1DGradientTest, InputGradientCheck)
{
	NEURAL_NETWORK::Convolution1D conv(1, 2, 3, 1, 0, 1); 

	conv.forward(test_input, true);
	auto output = conv.GetOutput();

	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	conv.backward(d_output);

	const auto& d_input = conv.GetDInput();

	EXPECT_EQ(d_input.rows(), test_input.rows());
	EXPECT_EQ(d_input.cols(), test_input.cols());

	bool has_nonzero = false;
	for (int i = 0; i < d_input.rows(); i++)
	{
		for (int j = 0; j < d_input.cols(); j++)
		{
			if (std::abs(d_input(i, j)) > 1e-8)
			{
				has_nonzero = true;
				break;
			}
		}
		if (has_nonzero) break;
	}
	EXPECT_TRUE(has_nonzero) << "Input gradients are all zero";
}

TEST_F(Convolution1DGradientTest, RegularizationGradientCheck)
{
	NEURAL_NETWORK::Convolution1D conv_reg(2, 2, 3, 1, 0, 1, 0.0, 0.1, 0.0, 0.1);
	NEURAL_NETWORK::Convolution1D conv_no_reg(2, 2, 3, 1, 0, 1);

	auto [weights, biases] = conv_no_reg.GetParameters();
	weights.setConstant(0.5);  
	biases.setConstant(0.2);   

	conv_reg.SetParameters(weights, biases);
	conv_no_reg.SetParameters(weights, biases);

	conv_reg.forward(test_input, true);
	conv_no_reg.forward(test_input, true);

	auto output = conv_reg.GetOutput();
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());

	conv_reg.backward(d_output);
	conv_no_reg.backward(d_output);

	const auto& d_weights_reg = conv_reg.GetDWeights();
	const auto& d_weights_no_reg = conv_no_reg.GetDWeights();

	const auto& d_biases_reg = conv_reg.GetDBiases();
	const auto& d_biases_no_reg = conv_no_reg.GetDBiases();

	double weight_diff_sum = 0.0;
	double bias_diff_sum = 0.0;

	for (int i = 0; i < d_weights_reg.size(); i++)
	{
		weight_diff_sum += std::abs(d_weights_reg.data()[i] - d_weights_no_reg.data()[i]);
	}

	for (int i = 0; i < d_biases_reg.size(); i++)
	{
		bias_diff_sum += std::abs(d_biases_reg.data()[i] - d_biases_no_reg.data()[i]);
	}

	EXPECT_GT(weight_diff_sum, 1e-6) << "L2 regularization should affect weight gradients";
	EXPECT_GT(bias_diff_sum, 1e-6) << "L2 regularization should affect bias gradients";
}


TEST_F(Convolution1DGradientTest, GradientFlowWithStride)
{
	NEURAL_NETWORK::Convolution1D conv_stride1(2, 2, 3, 1, 0, 1);
	
	NEURAL_NETWORK::Convolution1D conv_stride2(2, 2, 3, 1, 1, 2);

	conv_stride1.forward(test_input, true);
	conv_stride2.forward(test_input, true);

	auto output1 = conv_stride1.GetOutput();
	auto output2 = conv_stride2.GetOutput();

	Eigen::MatrixXd d_output1 = Eigen::MatrixXd::Ones(output1.rows(), output1.cols());
	Eigen::MatrixXd d_output2 = Eigen::MatrixXd::Ones(output2.rows(), output2.cols());

	conv_stride1.backward(d_output1);
	conv_stride2.backward(d_output2);

	const auto& d_input1 = conv_stride1.GetDInput();
	const auto& d_input2 = conv_stride2.GetDInput();

	EXPECT_EQ(d_input1.rows(), test_input.rows());
	EXPECT_EQ(d_input1.cols(), test_input.cols());
	EXPECT_EQ(d_input2.rows(), test_input.rows());
	EXPECT_EQ(d_input2.cols(), test_input.cols());

	EXPECT_GT(d_input1.cwiseAbs().maxCoeff(), 1e-8);
	EXPECT_GT(d_input2.cwiseAbs().maxCoeff(), 1e-8);
}


TEST_F(Convolution1DGradientTest, GradientFlowWithPadding)
{
	NEURAL_NETWORK::Convolution1D conv_no_pad(2, 2, 3, 1, 0, 1);
	NEURAL_NETWORK::Convolution1D conv_pad(2, 2, 3, 1, 1, 1);

	conv_no_pad.forward(test_input, true);
	conv_pad.forward(test_input, true);

	auto output1 = conv_no_pad.GetOutput();
	auto output2 = conv_pad.GetOutput();

	Eigen::MatrixXd d_output1 = Eigen::MatrixXd::Ones(output1.rows(), output1.cols());
	Eigen::MatrixXd d_output2 = Eigen::MatrixXd::Ones(output2.rows(), output2.cols());

	conv_no_pad.backward(d_output1);
	conv_pad.backward(d_output2);

	const auto& d_input1 = conv_no_pad.GetDInput();
	const auto& d_input2 = conv_pad.GetDInput();

	EXPECT_EQ(d_input1.rows(), test_input.rows());
	EXPECT_EQ(d_input1.cols(), test_input.cols());
	EXPECT_EQ(d_input2.rows(), test_input.rows());
	EXPECT_EQ(d_input2.cols(), test_input.cols());

	EXPECT_GT(d_input1.cwiseAbs().maxCoeff(), 1e-8);
	EXPECT_GT(d_input2.cwiseAbs().maxCoeff(), 1e-8);

	SUCCEED() << "Padding gradient flow test completed successfully";
}