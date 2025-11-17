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
        // Small signal for gradient checking: batch=2, length=3, channels=1 -> (2, 3)
        test_input = Eigen::MatrixXd(2, 3);
        test_input << 1.0, 2.0, 3.0,
                     0.5, 1.5, 2.5;

        epsilon = 1e-5;
        tolerance = 1e-3; // Relaxed tolerance for numerical gradient checking
    }

    // Numerical gradient checking helper function
    double computeNumericalGradient(NEURAL_NETWORK::Convolution1D& conv,
                                   const Eigen::MatrixXd& input,
                                   const Eigen::MatrixXd& target,
                                   int param_idx, bool is_weight = true)
    {
        // Get original parameters
        auto [weights, biases] = conv.GetParameters();

        // Forward pass to get original loss
        conv.forward(input, true);
        auto output = conv.GetOutput();
        double loss_original = 0.5 * (output - target).array().square().sum();

        // Perturb parameter
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

        // Forward pass with perturbed parameter
        conv.forward(input, true);
        auto output_perturbed = conv.GetOutput();
        double loss_perturbed = 0.5 * (output_perturbed - target).array().square().sum();

        // Restore original parameters
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

// Test weight gradients using numerical differentiation
TEST_F(Convolution1DGradientTest, WeightGradientCheck)
{
    NEURAL_NETWORK::Convolution1D conv(2, 2, 3, 1, 0, 1); // 2 filters, length 2, input length 3

    // Create simple target output
    conv.forward(test_input, true);
    auto output_size = conv.GetOutput();
    Eigen::MatrixXd target = Eigen::MatrixXd::Ones(output_size.rows(), output_size.cols());

    // Compute analytical gradients
    conv.forward(test_input, true);
    auto output = conv.GetOutput();
    Eigen::MatrixXd d_output = output - target; // MSE gradient
    conv.backward(d_output);

    const auto& analytical_grads = conv.GetDWeights();

    // Check a subset of weight gradients numerically (just first few)
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

// Test bias gradients using numerical differentiation
TEST_F(Convolution1DGradientTest, BiasGradientCheck)
{
    NEURAL_NETWORK::Convolution1D conv(2, 2, 3, 1, 0, 1); // 2 filters

    // Create target output
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

    // Compute analytical gradients
    conv.forward(test_input, true);
    auto output = conv.GetOutput();
    Eigen::MatrixXd d_output = output - target; // MSE gradient
    conv.backward(d_output);

    const auto& analytical_grads = conv.GetDBiases();

    // Check all bias gradients numerically
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

// Test input gradients (for completeness)
TEST_F(Convolution1DGradientTest, InputGradientCheck)
{
    NEURAL_NETWORK::Convolution1D conv(1, 2, 3, 1, 0, 1); // Simple case

    conv.forward(test_input, true);
    auto output = conv.GetOutput();

    // Use simple gradient for testing
    Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
    conv.backward(d_output);

    const auto& d_input = conv.GetDInput();

    // Check that input gradients have correct dimensions
    EXPECT_EQ(d_input.rows(), test_input.rows());
    EXPECT_EQ(d_input.cols(), test_input.cols());

    // Check that input gradients are not all zero
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

// Test that regularization affects gradients correctly
TEST_F(Convolution1DGradientTest, RegularizationGradientCheck)
{
    // Test with L2 regularization
    NEURAL_NETWORK::Convolution1D conv_reg(2, 2, 3, 1, 0, 1, 0.0, 0.1, 0.0, 0.1);
    NEURAL_NETWORK::Convolution1D conv_no_reg(2, 2, 3, 1, 0, 1);

    // Set same non-zero parameters for both (regularization needs non-zero weights/biases)
    auto [weights, biases] = conv_no_reg.GetParameters();
    weights.setConstant(0.5);  // Set non-zero weights
    biases.setConstant(0.2);   // Set non-zero biases

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

    // With L2 regularization, gradients should be different (larger in magnitude)
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

// Test gradient flow through different stride values
TEST_F(Convolution1DGradientTest, GradientFlowWithStride)
{
    // Use input that matches our test_input dimensions (2, 3)
    // Test with stride = 1
    NEURAL_NETWORK::Convolution1D conv_stride1(2, 2, 3, 1, 0, 1);
    // Test with stride = 2 - need to use padding to avoid negative dimensions
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

    // Both should have same input gradient dimensions
    EXPECT_EQ(d_input1.rows(), test_input.rows());
    EXPECT_EQ(d_input1.cols(), test_input.cols());
    EXPECT_EQ(d_input2.rows(), test_input.rows());
    EXPECT_EQ(d_input2.cols(), test_input.cols());

    // Gradients should be non-zero for both
    EXPECT_GT(d_input1.cwiseAbs().maxCoeff(), 1e-8);
    EXPECT_GT(d_input2.cwiseAbs().maxCoeff(), 1e-8);
}

// Test gradient flow through padding
TEST_F(Convolution1DGradientTest, GradientFlowWithPadding)
{
    // Use input that matches our test_input dimensions (2, 3)
    // Test with padding = 0
    NEURAL_NETWORK::Convolution1D conv_no_pad(2, 2, 3, 1, 0, 1);
    // Test with padding = 1
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

    // Both should have same input gradient dimensions (original input size)
    EXPECT_EQ(d_input1.rows(), test_input.rows());
    EXPECT_EQ(d_input1.cols(), test_input.cols());
    EXPECT_EQ(d_input2.rows(), test_input.rows());
    EXPECT_EQ(d_input2.cols(), test_input.cols());

    // Gradients should be non-zero for both
    EXPECT_GT(d_input1.cwiseAbs().maxCoeff(), 1e-8);
    EXPECT_GT(d_input2.cwiseAbs().maxCoeff(), 1e-8);

    // Note: In this simple test case, padding may not significantly affect input gradients
    // This is a basic functionality test to ensure no crashes occur
    SUCCEED() << "Padding gradient flow test completed successfully";
}