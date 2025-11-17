#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "Convolution1D.h"

class Convolution1DTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Simple 1D signal: batch_size=1, length=8, channels=2 (I/Q)
        test_input_simple = Eigen::MatrixXd(1, 16);
        test_input_simple << 1, 0.5, 2, 1.0, 3, 1.5, 4, 2.0, 5, 2.5, 6, 3.0, 7, 3.5, 8, 4.0;

        // Batch of 2 signals, length=6, channels=1
        test_input_batch = Eigen::MatrixXd(2, 6);
        test_input_batch << 1, 2, 3, 4, 5, 6,
                           6, 5, 4, 3, 2, 1;

        // Multi-channel signal: batch_size=1, length=4, channels=3
        test_input_multichannel = Eigen::MatrixXd(1, 12);
        test_input_multichannel << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

        // Small test signal for gradient checking
        test_input_small = Eigen::MatrixXd(1, 4);
        test_input_small << 1, 2, 3, 4;
    }

    Eigen::MatrixXd test_input_simple;
    Eigen::MatrixXd test_input_batch;
    Eigen::MatrixXd test_input_multichannel;
    Eigen::MatrixXd test_input_small;
    const double tolerance = 1e-6;
};

// Test constructor and basic initialization
TEST_F(Convolution1DTest, ConstructorInitialization)
{
    NEURAL_NETWORK::Convolution1D conv(16, 3, 8, 2, 1, 1);

    // Check tensor dimensions
    const auto& weights = conv.GetWeightsTensor();
    const auto& biases = conv.GetBiasesVector();

    EXPECT_EQ(weights.dimension(0), 3);  // filter_length
    EXPECT_EQ(weights.dimension(1), 2);  // input_channels
    EXPECT_EQ(weights.dimension(2), 16); // number_of_filters

    EXPECT_EQ(biases.size(), 16);

    // Check biases are initialized to zero
    for (int i = 0; i < biases.size(); i++)
    {
        EXPECT_NEAR(biases(i), 0.0, tolerance);
    }

    // Check He initialization bounds (weights should be reasonable)
    double max_weight = 0.0;
    for (int i = 0; i < weights.size(); i++)
    {
        max_weight = std::max(max_weight, std::abs(weights.data()[i]));
    }
    EXPECT_GT(max_weight, 0.0);
    EXPECT_LT(max_weight, 2.0); // Reasonable He initialization bound
}

// Test basic forward pass output dimensions
TEST_F(Convolution1DTest, ForwardPassOutputDimensions)
{
    NEURAL_NETWORK::Convolution1D conv(4, 3, 8, 2, 0, 1); // no padding

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    // Output length = (8 - 3) + 1 = 6
    EXPECT_EQ(output.rows(), 1);          // batch_size
    EXPECT_EQ(output.cols(), 6 * 4);      // output_length * num_filters = 6 * 4 = 24
}

// Test forward pass with padding
TEST_F(Convolution1DTest, ForwardPassWithPadding)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 8, 2, 1, 1); // padding=1

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    // With padding: output_length = (8 + 2*1 - 3) + 1 = 8
    EXPECT_EQ(output.rows(), 1);          // batch_size
    EXPECT_EQ(output.cols(), 8 * 2);      // output_length * num_filters = 8 * 2 = 16
}

// Test forward pass with stride
TEST_F(Convolution1DTest, ForwardPassWithStride)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 8, 2, 0, 2); // stride=2

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    // With stride=2: output_length = (8 - 3) / 2 + 1 = 3
    EXPECT_EQ(output.rows(), 1);          // batch_size
    EXPECT_EQ(output.cols(), 3 * 2);      // output_length * num_filters = 3 * 2 = 6
}

// Test batch processing
TEST_F(Convolution1DTest, BatchProcessing)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 6, 1, 0, 1);

    conv.forward(test_input_batch, false);
    const auto& output = conv.GetOutput();

    // Output length = (6 - 3) + 1 = 4
    EXPECT_EQ(output.rows(), 2);          // batch_size
    EXPECT_EQ(output.cols(), 4 * 2);      // output_length * num_filters = 4 * 2 = 8

    // Check that different inputs produce different outputs
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

// Test multi-channel input processing
TEST_F(Convolution1DTest, MultiChannelInput)
{
    NEURAL_NETWORK::Convolution1D conv(3, 2, 4, 3, 0, 1); // 3 input channels

    conv.forward(test_input_multichannel, false);
    const auto& output = conv.GetOutput();

    // Output length = (4 - 2) + 1 = 3
    EXPECT_EQ(output.rows(), 1);          // batch_size
    EXPECT_EQ(output.cols(), 3 * 3);      // output_length * num_filters = 3 * 3 = 9
}

// Test output numerical stability (no NaN or Inf)
TEST_F(Convolution1DTest, OutputNumericalStability)
{
    NEURAL_NETWORK::Convolution1D conv(4, 3, 8, 2, 1, 1);

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            EXPECT_FALSE(std::isnan(output(i, j))) << "NaN found at (" << i << "," << j << ")";
            EXPECT_FALSE(std::isinf(output(i, j))) << "Inf found at (" << i << "," << j << ")";
        }
    }
}

// Test that predictions match output
TEST_F(Convolution1DTest, PredictionsMatchOutput)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 8, 2, 1, 1);

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();
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

// Test backward pass gradient shapes
TEST_F(Convolution1DTest, BackwardPassGradientShapes)
{
    NEURAL_NETWORK::Convolution1D conv(3, 3, 8, 2, 0, 1);

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

    conv.backward(d_values);

    // Check gradient dimensions
    const auto& d_weights = conv.GetDWeights();
    const auto& d_biases = conv.GetDBiases();
    const auto& d_input = conv.GetDInput();

    // d_weights should match weight dimensions
    const auto& weights = conv.GetWeights();
    EXPECT_EQ(d_weights.rows(), weights.rows());
    EXPECT_EQ(d_weights.cols(), weights.cols());

    // d_biases should match bias dimensions
    const auto& biases = conv.GetBiases();
    EXPECT_EQ(d_biases.size(), biases.size());

    // d_input should match input dimensions
    EXPECT_EQ(d_input.rows(), test_input_simple.rows());
    EXPECT_EQ(d_input.cols(), test_input_simple.cols());
}

// Test backward pass produces non-zero gradients
TEST_F(Convolution1DTest, BackwardPassNonZeroGradients)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 4, 1, 0, 1);

    conv.forward(test_input_small, false);
    const auto& output = conv.GetOutput();

    Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

    conv.backward(d_values);

    const auto& d_weights = conv.GetDWeights();
    const auto& d_biases = conv.GetDBiases();
    const auto& d_input = conv.GetDInput();

    // Check that we have non-zero gradients
    bool has_nonzero_weight_grad = d_weights.cwiseAbs().maxCoeff() > tolerance;
    bool has_nonzero_bias_grad = d_biases.cwiseAbs().maxCoeff() > tolerance;
    bool has_nonzero_input_grad = d_input.cwiseAbs().maxCoeff() > tolerance;

    EXPECT_TRUE(has_nonzero_weight_grad) << "Weight gradients are all zero";
    EXPECT_TRUE(has_nonzero_bias_grad) << "Bias gradients are all zero";
    EXPECT_TRUE(has_nonzero_input_grad) << "Input gradients are all zero";
}

// Test backward pass numerical stability
TEST_F(Convolution1DTest, BackwardPassNumericalStability)
{
    NEURAL_NETWORK::Convolution1D conv(3, 3, 8, 2, 1, 1);

    conv.forward(test_input_simple, false);
    const auto& output = conv.GetOutput();

    Eigen::MatrixXd d_values = Eigen::MatrixXd::Random(output.rows(), output.cols());

    conv.backward(d_values);

    const auto& d_weights = conv.GetDWeights();
    const auto& d_biases = conv.GetDBiases();
    const auto& d_input = conv.GetDInput();

    // Check for NaN/Inf in gradients
    for (int i = 0; i < d_weights.rows(); i++)
    {
        for (int j = 0; j < d_weights.cols(); j++)
        {
            EXPECT_FALSE(std::isnan(d_weights(i, j))) << "NaN in weight gradient at (" << i << "," << j << ")";
            EXPECT_FALSE(std::isinf(d_weights(i, j))) << "Inf in weight gradient at (" << i << "," << j << ")";
        }
    }

    for (int i = 0; i < d_biases.size(); i++)
    {
        EXPECT_FALSE(std::isnan(d_biases(i))) << "NaN in bias gradient at " << i;
        EXPECT_FALSE(std::isinf(d_biases(i))) << "Inf in bias gradient at " << i;
    }

    for (int i = 0; i < d_input.rows(); i++)
    {
        for (int j = 0; j < d_input.cols(); j++)
        {
            EXPECT_FALSE(std::isnan(d_input(i, j))) << "NaN in input gradient at (" << i << "," << j << ")";
            EXPECT_FALSE(std::isinf(d_input(i, j))) << "Inf in input gradient at (" << i << "," << j << ")";
        }
    }
}

// Test backward pass with batch processing
TEST_F(Convolution1DTest, BackwardPassWithBatches)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 6, 1, 0, 1);

    conv.forward(test_input_batch, false);
    const auto& output = conv.GetOutput();

    Eigen::MatrixXd d_values = Eigen::MatrixXd::Ones(output.rows(), output.cols());

    conv.backward(d_values);
    const auto& d_input = conv.GetDInput();

    EXPECT_EQ(d_input.rows(), test_input_batch.rows());
    EXPECT_EQ(d_input.cols(), test_input_batch.cols());

    // Check that each batch has non-zero gradients
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

// Test regularization parameters storage
TEST_F(Convolution1DTest, RegularizationParametersStored)
{
    NEURAL_NETWORK::Convolution1D conv(4, 3, 8, 2, 1, 1, 0.1, 0.2, 0.05, 0.15);

    EXPECT_NEAR(conv.GetWeightRegularizerL1(), 0.1, tolerance);
    EXPECT_NEAR(conv.GetWeightRegularizerL2(), 0.2, tolerance);
    EXPECT_NEAR(conv.GetBiasRegularizerL1(), 0.05, tolerance);
    EXPECT_NEAR(conv.GetBiasRegularizerL2(), 0.15, tolerance);
}

// Test regularization in backward pass
TEST_F(Convolution1DTest, RegularizationInBackwardPass)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 8, 2, 1, 1, 0.01, 0.01, 0.01, 0.01);

    // Forward and backward pass should work with regularization
    EXPECT_NO_THROW(conv.forward(test_input_simple, true));

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

// Test parameter get/set functionality
TEST_F(Convolution1DTest, ParameterGetSet)
{
    NEURAL_NETWORK::Convolution1D conv(2, 3, 4, 1, 0, 1);

    // Get original parameters
    auto [orig_weights, orig_biases] = conv.GetParameters();

    // Create new parameters
    Eigen::MatrixXd new_weights = Eigen::MatrixXd::Constant(orig_weights.rows(), orig_weights.cols(), 0.5);
    Eigen::RowVectorXd new_biases = Eigen::RowVectorXd::Constant(orig_biases.size(), 0.1);

    // Set new parameters
    conv.SetParameters(new_weights, new_biases);

    // Verify parameters were set correctly
    const auto& weights = conv.GetWeights();
    const auto& biases = conv.GetBiases();

    EXPECT_TRUE(weights.isApprox(new_weights, tolerance));
    EXPECT_TRUE(biases.isApprox(new_biases, tolerance));
}

// Test getter methods for layer properties
TEST_F(Convolution1DTest, LayerPropertyGetters)
{
    int num_filters = 8;
    int filter_length = 5;
    int input_length = 16;
    int input_channels = 3;
    int padding = 1;
    int stride = 2;

    NEURAL_NETWORK::Convolution1D conv(num_filters, filter_length, input_length,
                                       input_channels, padding, stride);

    EXPECT_EQ(conv.GetNumberOfFilters(), num_filters);
    EXPECT_EQ(conv.GetFilterLength(), filter_length);
    EXPECT_EQ(conv.GetInputLength(), input_length);
    EXPECT_EQ(conv.GetInputChannels(), input_channels);
    EXPECT_EQ(conv.GetPadding(), padding);
    EXPECT_EQ(conv.GetStride(), stride);
}

// Test simple mathematical correctness for a known case
TEST_F(Convolution1DTest, SimpleMathematicalCorrectness)
{
    // Create a simple case we can verify by hand
    NEURAL_NETWORK::Convolution1D conv(1, 2, 3, 1, 0, 1); // 1 filter, length 2, input length 3, 1 channel

    // Set known weights and biases
    Eigen::MatrixXd weights(1, 2);  // 1 filter, 2 elements
    weights << 1.0, 2.0;  // Simple filter [1, 2]

    Eigen::RowVectorXd biases(1);
    biases << 0.0;  // No bias

    conv.SetParameters(weights, biases);

    // Input signal: [1, 2, 3] (length 3, 1 channel)
    Eigen::MatrixXd input(1, 3);
    input << 1.0, 2.0, 3.0;

    conv.forward(input, false);
    const auto& output = conv.GetOutput();

    // Expected: [1*1 + 2*2, 2*1 + 3*2] = [5, 8]
    // Output should be (1, 2) - 1 batch, 2 output values
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 2);

    EXPECT_NEAR(output(0, 0), 5.0, tolerance);
    EXPECT_NEAR(output(0, 1), 8.0, tolerance);
}

// Test consistency of repeated forward passes
TEST_F(Convolution1DTest, ForwardPassConsistency)
{
    NEURAL_NETWORK::Convolution1D conv(3, 3, 8, 2, 1, 1);

    conv.forward(test_input_simple, false);
    const auto output1 = conv.GetOutput();

    conv.forward(test_input_simple, false);
    const auto output2 = conv.GetOutput();

    // Results should be identical
    EXPECT_TRUE(output1.isApprox(output2, tolerance));
}

// Test single filter case (edge case)
TEST_F(Convolution1DTest, SingleFilterCase)
{
    NEURAL_NETWORK::Convolution1D conv(1, 1, 4, 1, 0, 1); // 1x1 filter

    conv.forward(test_input_small, false);
    const auto& output = conv.GetOutput();

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 4); // Same length as input for 1x1 filter
}