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
        epsilon = 1e-5;
        tolerance = 1e-3; // Relaxed tolerance for gradient checking
        
        // Small test inputs for faster gradient checking
        test_input_2x2 = Eigen::MatrixXd(1, 4);
        test_input_2x2 << 0.1, 0.2, 0.3, 0.4;
        
        test_input_3x3 = Eigen::MatrixXd(1, 9);
        test_input_3x3 << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
        
        // Batch input for testing
        test_batch_2x2 = Eigen::MatrixXd(2, 4);
        test_batch_2x2 << 0.1, 0.2, 0.3, 0.4,
                          0.5, 0.6, 0.7, 0.8;
    }
    
    Eigen::MatrixXd test_input_2x2;
    Eigen::MatrixXd test_input_3x3;
    Eigen::MatrixXd test_batch_2x2;
    double epsilon;
    double tolerance;
};

TEST_F(GradientCheckTest, ConvolutionWeightGradients)
{
    // Test gradient checking for convolution layer weights
    NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1);
    
    // Forward pass
    conv.forward(test_input_2x2, true);
    
    // Create simple loss: sum of squares of output
    const auto& output = conv.GetOutput();
    double original_loss = 0.5 * output.array().square().sum();
    
    // Create gradient of loss w.r.t. output (derivative of 0.5*sum(output^2))
    Eigen::MatrixXd d_output = output; // d_loss/d_output = output
    
    // Backward pass to get analytical gradients
    conv.backward(d_output);
    const auto& analytical_d_weights = conv.GetDWeightsTensor();
    
    // Check gradients numerically for each weight
    auto weights = conv.GetWeightsTensor();
    
    for (int i = 0; i < weights.dimension(0); ++i) {
        for (int j = 0; j < weights.dimension(1); ++j) {
            for (int k = 0; k < weights.dimension(2); ++k) {
                for (int l = 0; l < weights.dimension(3); ++l) {
                    double original_weight = weights(i, j, k, l);
                    
                    // Define loss function for this specific weight
                    auto loss_func = [&](double w) -> double {
                        // Set the weight
                        auto temp_weights = weights;
                        temp_weights(i, j, k, l) = w;
                        conv.SetWeightsTensor(temp_weights);
                        
                        // Forward pass
                        conv.forward(test_input_2x2, false);
                        const auto& temp_output = conv.GetOutput();
                        
                        // Return loss
                        return 0.5 * temp_output.array().square().sum();
                    };
                    
                    // Calculate numerical gradient
                    double numerical_grad = NumericalGradient(loss_func, original_weight);
                    double analytical_grad = analytical_d_weights(i, j, k, l);
                    
                    // Check if gradients match
                    EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
                        << "Weight gradient mismatch at (" << i << "," << j << "," << k << "," << l << ")"
                        << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
                    
                    // Restore original weight
                    weights(i, j, k, l) = original_weight;
                    conv.SetWeightsTensor(weights);
                }
            }
        }
    }
}

TEST_F(GradientCheckTest, ConvolutionBiasGradients)
{
    // Test gradient checking for convolution layer biases
    NEURAL_NETWORK::Convolution conv(2, 2, 2, 2, 2, 1, false, 1, 1);
    
    // Forward pass
    conv.forward(test_input_2x2, true);
    
    // Create gradient of loss w.r.t. output
    const auto& output = conv.GetOutput();
    Eigen::MatrixXd d_output = output; // Simple quadratic loss gradient
    
    // Backward pass to get analytical gradients
    conv.backward(d_output);
    const auto& analytical_d_biases = conv.GetDBiasesVector();
    
    // Check gradients numerically for each bias
    auto biases = conv.GetBiasesVector();
    
    for (int i = 0; i < biases.size(); ++i) {
        double original_bias = biases(i);
        
        // Define loss function for this specific bias
        auto loss_func = [&](double b) -> double {
            // Set the bias
            auto temp_biases = biases;
            temp_biases(i) = b;
            conv.SetBiasesVector(temp_biases);
            
            // Forward pass
            conv.forward(test_input_2x2, false);
            const auto& temp_output = conv.GetOutput();
            
            // Return loss
            return 0.5 * temp_output.array().square().sum();
        };
        
        // Calculate numerical gradient
        double numerical_grad = NumericalGradient(loss_func, original_bias);
        double analytical_grad = analytical_d_biases(i);
        
        // Check if gradients match
        EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
            << "Bias gradient mismatch at index " << i
            << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
        
        // Restore original bias
        biases(i) = original_bias;
        conv.SetBiasesVector(biases);
    }
}

TEST_F(GradientCheckTest, ConvolutionInputGradients)
{
    // Test gradient checking for convolution input gradients
    NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);
    
    // Forward pass
    conv.forward(test_input_3x3, true);
    
    // Create gradient of loss w.r.t. output
    const auto& output = conv.GetOutput();
    Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
    
    // Backward pass to get analytical gradients
    conv.backward(d_output);
    const auto& analytical_d_input = conv.GetDInput();
    
    // Check gradients numerically for each input element
    auto input = test_input_3x3;
    
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            double original_input = input(i, j);
            
            // Define loss function for this specific input element
            auto loss_func = [&](double x) -> double {
                // Set the input element
                auto temp_input = input;
                temp_input(i, j) = x;
                
                // Forward pass
                conv.forward(temp_input, false);
                const auto& temp_output = conv.GetOutput();
                
                // Return loss (sum of outputs)
                return temp_output.sum();
            };
            
            // Calculate numerical gradient
            double numerical_grad = NumericalGradient(loss_func, original_input);
            double analytical_grad = analytical_d_input(i, j);
            
            // Check if gradients match
            EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
                << "Input gradient mismatch at (" << i << "," << j << ")"
                << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
        }
    }
}

TEST_F(GradientCheckTest, MaxPoolingInputGradients)
{
    // Test gradient checking for MaxPooling input gradients
    NEURAL_NETWORK::MaxPooling pool(1, 2, 2, 2, 1, 1);
    
    // Forward pass
    pool.forward(test_input_2x2, true);
    
    // Create gradient of loss w.r.t. output
    const auto& output = pool.GetOutput();
    Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
    
    // Backward pass to get analytical gradients
    pool.backward(d_output);
    const auto& analytical_d_input = pool.GetDInput();
    
    // Check gradients numerically for each input element
    auto input = test_input_2x2;
    
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            double original_input = input(i, j);
            
            // Define loss function for this specific input element
            auto loss_func = [&](double x) -> double {
                // Set the input element
                auto temp_input = input;
                temp_input(i, j) = x;
                
                // Forward pass
                pool.forward(temp_input, false);
                const auto& temp_output = pool.GetOutput();
                
                // Return loss (sum of outputs)
                return temp_output.sum();
            };
            
            // Calculate numerical gradient
            double numerical_grad = NumericalGradient(loss_func, original_input);
            double analytical_grad = analytical_d_input(i, j);
            
            // MaxPooling gradients are often zero (non-max elements) or one (max elements)
            // So we need more lenient tolerance for elements that should be zero
            double effective_tolerance = (std::abs(analytical_grad) < 1e-8) ? 1e-3 : tolerance;
            
            EXPECT_NEAR(numerical_grad, analytical_grad, effective_tolerance)
                << "MaxPooling input gradient mismatch at (" << i << "," << j << ")"
                << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
        }
    }
}

TEST_F(GradientCheckTest, LayerDenseWeightGradients)
{
    // Test gradient checking for dense layer (for comparison/validation)
    NEURAL_NETWORK::LayerDense dense(4, 2);
    
    // Forward pass
    dense.forward(test_input_2x2, true);
    
    // Create gradient of loss w.r.t. output
    const auto& output = dense.GetOutput();
    Eigen::MatrixXd d_output = output; // Quadratic loss gradient
    
    // Backward pass to get analytical gradients
    dense.backward(d_output);
    const auto& analytical_d_weights = dense.GetDWeights();
    
    // Check gradients numerically for each weight
    auto weights = dense.GetWeights();
    
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            double original_weight = weights(i, j);
            
            // Define loss function for this specific weight
            auto loss_func = [&](double w) -> double {
                // Set the weight
                auto temp_weights = weights;
                temp_weights(i, j) = w;
                dense.SetWeights(temp_weights);
                
                // Forward pass
                dense.forward(test_input_2x2, false);
                const auto& temp_output = dense.GetOutput();
                
                // Return loss
                return 0.5 * temp_output.array().square().sum();
            };
            
            // Calculate numerical gradient
            double numerical_grad = NumericalGradient(loss_func, original_weight);
            double analytical_grad = analytical_d_weights(i, j);
            
            // Check if gradients match
            EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
                << "Dense weight gradient mismatch at (" << i << "," << j << ")"
                << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
            
            // Restore original weight
            weights(i, j) = original_weight;
            dense.SetWeights(weights);
        }
    }
}

TEST_F(GradientCheckTest, ConvolutionWithRegularization)
{
    // Test gradient checking for convolution with regularization
    NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1, 0.1, 0.1); // L1=0.1, L2=0.1
    
    // Set known weights for predictable regularization
    auto weights = conv.GetWeightsTensor();
    for (int i = 0; i < weights.dimension(0); ++i) {
        for (int j = 0; j < weights.dimension(1); ++j) {
            for (int k = 0; k < weights.dimension(2); ++k) {
                for (int l = 0; l < weights.dimension(3); ++l) {
                    weights(i, j, k, l) = 0.5 * (i + j + k + l + 1);
                }
            }
        }
    }
    conv.SetWeightsTensor(weights);
    
    // Forward pass
    conv.forward(test_input_2x2, true);
    
    // Create gradient with regularization included
    const auto& output = conv.GetOutput();
    Eigen::MatrixXd d_output = output; // Data loss gradient
    
    // Backward pass (includes regularization)
    conv.backward(d_output);
    const auto& analytical_d_weights = conv.GetDWeightsTensor();
    
    // Check gradients numerically including regularization
    for (int i = 0; i < weights.dimension(0); ++i) {
        for (int j = 0; j < weights.dimension(1); ++j) {
            for (int k = 0; k < weights.dimension(2); ++k) {
                for (int l = 0; l < weights.dimension(3); ++l) {
                    double original_weight = weights(i, j, k, l);
                    
                    // Define loss function including regularization
                    auto loss_func = [&](double w) -> double {
                        // Set the weight
                        auto temp_weights = weights;
                        temp_weights(i, j, k, l) = w;
                        conv.SetWeightsTensor(temp_weights);
                        
                        // Forward pass
                        conv.forward(test_input_2x2, false);
                        const auto& temp_output = conv.GetOutput();
                        
                        // Data loss
                        double data_loss = 0.5 * temp_output.array().square().sum();
                        
                        // Regularization loss
                        double reg_loss = conv.CalculateRegularizationLoss();
                        
                        return data_loss + reg_loss;
                    };
                    
                    // Calculate numerical gradient
                    double numerical_grad = NumericalGradient(loss_func, original_weight);
                    double analytical_grad = analytical_d_weights(i, j, k, l);
                    
                    // Check if gradients match (more lenient tolerance due to regularization)
                    EXPECT_NEAR(numerical_grad, analytical_grad, tolerance * 10)
                        << "Regularized weight gradient mismatch at (" << i << "," << j << "," << k << "," << l << ")"
                        << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
                    
                    // Restore original weight
                    weights(i, j, k, l) = original_weight;
                    conv.SetWeightsTensor(weights);
                }
            }
        }
    }
}

TEST_F(GradientCheckTest, ConvolutionBatchGradients)
{
    // Test gradient checking with batch inputs
    NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1);
    
    // Forward pass with batch
    conv.forward(test_batch_2x2, true);
    
    // Create gradient of loss w.r.t. output
    const auto& output = conv.GetOutput();
    Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
    
    // Backward pass to get analytical gradients
    conv.backward(d_output);
    const auto& analytical_d_input = conv.GetDInput();
    
    // Check input gradients for batch processing
    auto input = test_batch_2x2;
    
    // Check a few input elements (not all to keep test time reasonable)
    std::vector<std::pair<int, int>> test_positions = {{0, 0}, {0, 3}, {1, 1}, {1, 2}};
    
    for (const auto& pos : test_positions) {
        int i = pos.first;
        int j = pos.second;
        double original_input = input(i, j);
        
        // Define loss function for this specific input element
        auto loss_func = [&](double x) -> double {
            // Set the input element
            auto temp_input = input;
            temp_input(i, j) = x;
            
            // Forward pass
            conv.forward(temp_input, false);
            const auto& temp_output = conv.GetOutput();
            
            // Return loss (sum of outputs)
            return temp_output.sum();
        };
        
        // Calculate numerical gradient
        double numerical_grad = NumericalGradient(loss_func, original_input);
        double analytical_grad = analytical_d_input(i, j);
        
        // Check if gradients match
        EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
            << "Batch input gradient mismatch at (" << i << "," << j << ")"
            << " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
    }
}