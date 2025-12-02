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
		CreateSyntheticDataset();
		tolerance = 1e-5;
	}
	
	void CreateSyntheticDataset()
	{
		int num_samples = 16;
		int image_size = 8 * 8; 
		int num_classes = 3;
		X_train = Eigen::MatrixXd::Random(num_samples, image_size);
		X_train = (X_train.array() + 1.0) / 2.0; 
		
		y_train = Eigen::MatrixXd::Zero(num_samples, num_classes);
		for (int i = 0; i < num_samples; ++i) {
			int class_id = i % num_classes;
			y_train(i, class_id) = 1.0;
		}
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
	NEURAL_NETWORK::Model model;
	EXPECT_NO_THROW({
		model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
		model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
			4, 3, 3, 8, 8, 1, false, 1, 1));
		model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
		model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
			16, 2, 6, 6, 4, 2));
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
	NEURAL_NETWORK::Model model;
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		2, 3, 3, 8, 8, 1, true, 1, 1)); 
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 8, 8, 2, 2)); 
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(32, 3));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>());
	
	model.Finalize();
	
	Eigen::MatrixXd predictions;
	EXPECT_NO_THROW(predictions = model.Predict(X_test, 2));
	
	EXPECT_EQ(predictions.rows(), X_test.rows());
	EXPECT_EQ(predictions.cols(), 3); 
	
	for (int i = 0; i < predictions.rows(); ++i) {
		double row_sum = predictions.row(i).sum();
		EXPECT_NEAR(row_sum, 1.0, tolerance) << "Row " << i << " doesn't sum to 1";
		for (int j = 0; j < predictions.cols(); ++j) {
			EXPECT_GE(predictions(i, j), 0.0) << "Negative probability at (" << i << "," << j << ")";
		}
	}
}

TEST_F(CNNIntegrationTest, CNNTraining)
{
	NEURAL_NETWORK::Model model;
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		4, 3, 3, 8, 8, 1, true, 1, 1));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 8, 8, 4, 2)); 
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.01)); 
	
	model.Finalize();
	
	Eigen::MatrixXd initial_predictions = model.Predict(X_test, 4);
	
	EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 3, 1, 0, X_test, y_test));
	
	Eigen::MatrixXd final_predictions = model.Predict(X_test, 4);
	
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
	NEURAL_NETWORK::Model model;
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		4, 3, 3, 8, 8, 1, true, 1, 1));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.25)); 
	
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 8, 8, 4, 2));
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5)); 
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>());
	
	model.Finalize();
	
	EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 2, 1, 0));
	
	Eigen::MatrixXd predictions;
	EXPECT_NO_THROW(predictions = model.Predict(X_test, 4));
	
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
	NEURAL_NETWORK::Model model;
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		4, 3, 3, 8, 8, 1, true, 1, 1, 0.01, 0.01)); 
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 8, 8, 4, 2));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 8, 0.01, 0.01)); 
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 3));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	
	model.Finalize();
	
	EXPECT_NO_THROW(model.Train(X_train, y_train, 8, 3, 1, 0));
	
	EXPECT_NO_THROW(model.Evaluate(X_test, y_test, 4));
}

TEST_F(CNNIntegrationTest, DeepCNNArchitecture)
{
	NEURAL_NETWORK::Model model;
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		8, 3, 3, 8, 8, 1, true, 1, 1));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 8, 8, 8, 2)); 
	
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		16, 3, 3, 4, 4, 8, true, 1, 1));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 2, 4, 4, 16, 2)); 
	
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 16));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(16, 3));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>());
	
	model.Finalize();
	
	EXPECT_NO_THROW(model.Train(X_train, y_train, 4, 2, 1, 0));
	EXPECT_NO_THROW(model.Predict(X_test, 4));
}

TEST_F(CNNIntegrationTest, CNNSaveLoad)
{
	std::string model_file = "test_cnn_integration.bin";
	
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
	
	model.Train(X_train, y_train, 8, 2, 1, 0);
	
	Eigen::MatrixXd original_predictions = model.Predict(X_test, 4);
	
	EXPECT_NO_THROW(model.SaveModel(model_file));
	
	NEURAL_NETWORK::Model loaded_model;
	EXPECT_NO_THROW(loaded_model.LoadModel(model_file));
	
	Eigen::MatrixXd loaded_predictions = loaded_model.Predict(X_test, 4);
	
	for (int i = 0; i < original_predictions.rows(); ++i) {
		for (int j = 0; j < original_predictions.cols(); ++j) {
			EXPECT_NEAR(original_predictions(i, j), loaded_predictions(i, j), tolerance)
				<< "Prediction mismatch after save/load at (" << i << "," << j << ")";
		}
	}
	
	if (std::filesystem::exists(model_file)) {
		std::filesystem::remove(model_file);
	}
}

TEST_F(CNNIntegrationTest, DifferentOptimizers)
{
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
		
		if (opt_name == "Adam") {
			model.Set(
				std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				std::make_unique<NEURAL_NETWORK::Adam>(0.01));
		} else if (opt_name == "SGD") {
			model.Set(
				std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				std::make_unique<NEURAL_NETWORK::SGD>(0.1)); 
		}
		
		model.Finalize();
		
		EXPECT_NO_THROW(model.Train(X_train, y_train, 8, 2, 1, 0))
			<< "Training failed with " << opt_name << " optimizer";
		
		Eigen::MatrixXd predictions;
		EXPECT_NO_THROW(predictions = model.Predict(X_test, 4))
			<< "Inference failed with " << opt_name << " optimizer";
		
		EXPECT_EQ(predictions.rows(), X_test.rows());
		EXPECT_EQ(predictions.cols(), 3);
	}
}

TEST_F(CNNIntegrationTest, BatchSizeVariations)
{
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
	
	std::vector<int> batch_sizes = {1, 2, 4, 8};
	
	for (int batch_size : batch_sizes) {
		EXPECT_NO_THROW(model.Train(X_train, y_train, batch_size, 1, 1, 0))
			<< "Training failed with batch size " << batch_size;
		
		EXPECT_NO_THROW(model.Predict(X_test, batch_size))
			<< "Inference failed with batch size " << batch_size;
	}
}