#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Model.h"
#include "LayerInput.h"
#include "Convolution2D.h"
#include "MaxPooling.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"
#include "AccuracyCategorical.h"
#include "Adam.h"
#include "StochasticGradientDescent.h"

class CNNIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create small synthetic dataset for testing
        CreateSyntheticDataset();
        tolerance = 1e-5;
    }
    
    void CreateSyntheticDataset()
    {
        // Create 16 samples of 8x8 images with 3 classes
        int num_samples = 16;
        int image_size = 8 * 8; // 8x8 images
        int num_classes = 3;
        
        X_train = Eigen::MatrixXd::Random(num_samples, image_size);
        X_train = (X_train.array() + 1.0) / 2.0; // Scale to [0,1]
        
        // Create targets based on simple patterns
        y_train = Eigen::MatrixXd::Zero(num_samples, num_classes);
        for (int i = 0; i < num_samples; ++i) {
            int class_id = i % num_classes;
            y_train(i, class_id) = 1.0;
        }
        
        // Create smaller test set
        int test_samples = 6;
        X_test = Eigen::MatrixXd::Random(test_samples, image_size);
        X_test = (X_test.array() + 1.0) / 2.0;
        
        y_test = Eigen::MatrixXd::Zero(test_samples, num_classes);
        for (int i = 0; i < test_samples; ++i) {
            int class_id = i % num_classes;
            y_test(i, class_id) = 1.0;
        }
    }
    
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    double tolerance;
};

TEST_F(CNNIntegrationTest, SimpleCNNConstruction)
{
    // Test basic CNN construction without errors
    NEURAL_NETWORK::Model model;
    
    EXPECT_NO_THROW({
        model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
        
        // Conv layer: 8x8x1 -> 6x6x4 (no padding, 3x3 kernel)
        model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
            4, 3, 3, 8, 8, 1, false, 1, 1));
        model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
        
        // MaxPool: 6x6x4 -> 3x3x4 (2x2 pool, stride 2)
        model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
            16, 2, 6, 6, 4, 2));
        
        // Dense layers: 3*3*4=36 -> 8 -> 3
        model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(36, 8));
        model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
        model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
        model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
        
        model.Set(
            std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
            std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
            std::make_unique<NEURAL_NETWORK::Adam>(0.001));
        
        model.Finalize();
    });
}

TEST_F(CNNIntegrationTest, CNNForwardPass)
{
    // Test that CNN can perform forward pass without errors
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        2, 3, 3, 8, 8, 1, true, 1, 1)); // with padding
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 2, 2)); // 8x8x2 -> 4x4x2
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(32, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>());
    
    model.Finalize();
    
    // Test forward pass
    Eigen::MatrixXd predictions;
    EXPECT_NO_THROW(predictions = model.Predict(X_test, 2));
    
    // Check output dimensions
    EXPECT_EQ(predictions.rows(), X_test.rows());
    EXPECT_EQ(predictions.cols(), 3); // 3 classes
    
    // Check that outputs are valid probabilities (sum to 1)
    for (int i = 0; i < predictions.rows(); ++i) {
        double row_sum = predictions.row(i).sum();
        EXPECT_NEAR(row_sum, 1.0, tolerance) << "Row " << i << " doesn't sum to 1";
        
        // Check all values are non-negative
        for (int j = 0; j < predictions.cols(); ++j) {
            EXPECT_GE(predictions(i, j), 0.0) << "Negative probability at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CNNIntegrationTest, CNNTraining)
{
    // Test that CNN can train without errors and improves
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        4, 3, 3, 8, 8, 1, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 4, 2)); // 8x8x4 -> 4x4x4
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.01)); // Higher LR for faster convergence
    
    model.Finalize();
    
    // Get initial predictions
    Eigen::MatrixXd initial_predictions = model.Predict(X_test, 4);
    
    // Train for a few epochs
    EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 3, 1, X_test, y_test));
    
    // Get final predictions
    Eigen::MatrixXd final_predictions = model.Predict(X_test, 4);
    
    // Predictions should have changed (model learned something)
    bool predictions_changed = false;
    for (int i = 0; i < initial_predictions.rows(); ++i) {
        for (int j = 0; j < initial_predictions.cols(); ++j) {
            if (std::abs(initial_predictions(i, j) - final_predictions(i, j)) > tolerance) {
                predictions_changed = true;
                break;
            }
        }
        if (predictions_changed) break;
    }
    
    EXPECT_TRUE(predictions_changed) << "Model predictions didn't change during training";
}

TEST_F(CNNIntegrationTest, CNNWithDropout)
{
    // Test CNN with dropout layers
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        4, 3, 3, 8, 8, 1, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.25)); // Dropout after conv
    
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 4, 2));
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5)); // Dropout after dense
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>());
    
    model.Finalize();
    
    // Test that training and inference work
    EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 2, 1));
    
    // Test that inference works (dropout should be disabled)
    Eigen::MatrixXd predictions;
    EXPECT_NO_THROW(predictions = model.Predict(X_test, 4));
    
    // Multiple inference calls should give same results (dropout disabled)
    Eigen::MatrixXd predictions2 = model.Predict(X_test, 4);
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            EXPECT_NEAR(predictions(i, j), predictions2(i, j), tolerance)
                << "Inference not deterministic at (" << i << "," << j << ")";
        }
    }
}

TEST_F(CNNIntegrationTest, CNNWithRegularization)
{
    // Test CNN with L1/L2 regularization
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        4, 3, 3, 8, 8, 1, true, 1, 1, 0.01, 0.01)); // L1=0.01, L2=0.01
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 4, 2));
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8, 0.01, 0.01)); // Regularized dense
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.01));
    
    model.Finalize();
    
    // Test training with regularization
    EXPECT_NO_THROW(model.Train(X_train, y_train, 8, 3, 1));
    
    // Test evaluation
    EXPECT_NO_THROW(model.Evaluate(X_test, y_test, 4));
}

TEST_F(CNNIntegrationTest, DeepCNNArchitecture)
{
    // Test deeper CNN architecture
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    
    // First conv block
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        8, 3, 3, 8, 8, 1, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 8, 2)); // 8x8x8 -> 4x4x8
    
    // Second conv block
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        16, 3, 3, 4, 4, 8, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 4, 4, 16, 2)); // 4x4x16 -> 2x2x16
    
    // Dense layers
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 16));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(16, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>());
    
    model.Finalize();
    
    // Test that deep architecture works
    EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 2, 1));
    EXPECT_NO_THROW(model.Predict(X_test, 4));
}

TEST_F(CNNIntegrationTest, CNNSaveLoad)
{
    // Test CNN save/load functionality
    std::string model_file = "test_cnn_integration.bin";
    
    // Create and train CNN
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        4, 3, 3, 8, 8, 1, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 4, 2));
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>());
    
    model.Finalize();
    
    // Train briefly
    model.Train(X_train, y_train, 8, 2, 1);
    
    // Get predictions before saving
    Eigen::MatrixXd original_predictions = model.Predict(X_test, 4);
    
    // Save model
    EXPECT_NO_THROW(model.SaveModel(model_file));
    
    // Load model
    NEURAL_NETWORK::Model loaded_model;
    EXPECT_NO_THROW(loaded_model.LoadModel(model_file));
    
    // Get predictions from loaded model
    Eigen::MatrixXd loaded_predictions = loaded_model.Predict(X_test, 4);
    
    // Predictions should be identical
    for (int i = 0; i < original_predictions.rows(); ++i) {
        for (int j = 0; j < original_predictions.cols(); ++j) {
            EXPECT_NEAR(original_predictions(i, j), loaded_predictions(i, j), tolerance)
                << "Prediction mismatch after save/load at (" << i << "," << j << ")";
        }
    }
    
    // Cleanup
    if (std::filesystem::exists(model_file)) {
        std::filesystem::remove(model_file);
    }
}

TEST_F(CNNIntegrationTest, DifferentOptimizers)
{
    // Test CNN with different optimizers
    std::vector<std::string> optimizer_names = {"Adam", "SGD"};
    
    for (const auto& opt_name : optimizer_names) {
        NEURAL_NETWORK::Model model;
        
        model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
        model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
            4, 3, 3, 8, 8, 1, true, 1, 1));
        model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
        model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
            16, 2, 8, 8, 4, 2));
        model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 3));
        model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
        
        // Set different optimizers
        if (opt_name == "Adam") {
            model.Set(
                std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
                std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
                std::make_unique<NEURAL_NETWORK::Adam>(0.01));
        } else if (opt_name == "SGD") {
            model.Set(
                std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
                std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
                std::make_unique<NEURAL_NETWORK::SGD>(0.1)); // Higher LR for SGD
        }
        
        model.Finalize();
        
        // Test training with different optimizers
        EXPECT_NO_THROW(model.Train(X_train, y_train, 8, 2, 1))
            << "Training failed with " << opt_name << " optimizer";
        
        // Test inference
        Eigen::MatrixXd predictions;
        EXPECT_NO_THROW(predictions = model.Predict(X_test, 4))
            << "Inference failed with " << opt_name << " optimizer";
        
        // Check prediction validity
        EXPECT_EQ(predictions.rows(), X_test.rows());
        EXPECT_EQ(predictions.cols(), 3);
    }
}

TEST_F(CNNIntegrationTest, BatchSizeVariations)
{
    // Test CNN with different batch sizes
    NEURAL_NETWORK::Model model;
    
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
        4, 3, 3, 8, 8, 1, true, 1, 1));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        16, 2, 8, 8, 4, 2));
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 3));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>());
    
    model.Finalize();
    
    // Test training with different batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8};
    
    for (int batch_size : batch_sizes) {
        EXPECT_NO_THROW(model.Train(X_train, y_train, batch_size, 1, 1))
            << "Training failed with batch size " << batch_size;
        
        // Test inference with different batch sizes
        EXPECT_NO_THROW(model.Predict(X_test, batch_size))
            << "Inference failed with batch size " << batch_size;
    }
}