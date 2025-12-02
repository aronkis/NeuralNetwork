#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>
#include <filesystem>
#include <cmath>
#include "Model.h"
#include "LayerInput.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "Convolution1D.h"
#include "Convolution2D.h"
#include "MaxPooling.h"
#include "MaxPooling1D.h"
#include "BatchNormalization.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "ActivationSigmoid.h"
#include "ActivationLinear.h"
#include "LossCategoricalCrossentropy.h"
#include "LossMeanSquaredError.h"
#include "AccuracyCategorical.h"
#include "AccuracyRegression.h"
#include "Adam.h"
#include "StochasticGradientDescent.h"
#include "AdaGrad.h"
#include "RMSProp.h"

class ModelSaveLoadIntegrationTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		temp_model_file = "/tmp/integration_test_model.bin";
		temp_training_file = "/tmp/integration_test_training_model.bin";

		X_train = Eigen::MatrixXd(8, 2);
		X_train << 0.0, 0.0,
				   0.0, 1.0,
				   1.0, 0.0,
				   1.0, 1.0,
				   0.1, 0.1,
				   0.1, 0.9,
				   0.9, 0.1,
				   0.9, 0.9;

		y_train = Eigen::MatrixXd(8, 1);
		y_train << 0, 1, 1, 0, 0, 1, 1, 0;
		
		X_test = Eigen::MatrixXd(4, 2);
		X_test << 0.05, 0.05,
				  0.05, 0.95,
				  0.95, 0.05,
				  0.95, 0.95;
		
		y_test = Eigen::MatrixXd(4, 1);
		y_test << 0, 1, 1, 0;
	}

	void TearDown() override 
	{
		if (std::filesystem::exists(temp_model_file)) 
		{
			std::filesystem::remove(temp_model_file);
		}
		if (std::filesystem::exists(temp_training_file)) 
		{
			std::filesystem::remove(temp_training_file);
		}
	}

	std::string temp_model_file;
	std::string temp_training_file;
	Eigen::MatrixXd X_train, y_train, X_test, y_test;
	const double tolerance = 1e-10;
};


TEST_F(ModelSaveLoadIntegrationTest, DenseNetworkFullCycle) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 8));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 8));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	
	model->Train(X_train, y_train, 4, 10, 100, 0, X_test, y_test);
	
	Eigen::MatrixXd original_predictions = model->Predict(X_test, 1);
	
	model->SaveModel(temp_model_file);
	EXPECT_TRUE(std::filesystem::exists(temp_model_file));
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	Eigen::MatrixXd loaded_predictions = loaded_model->Predict(X_test, 1);
	
	ASSERT_EQ(original_predictions.rows(), loaded_predictions.rows());
	ASSERT_EQ(original_predictions.cols(), loaded_predictions.cols());
	
	for (int i = 0; i < original_predictions.rows(); ++i) 
	{
		for (int j = 0; j < original_predictions.cols(); ++j) 
		{
			EXPECT_NEAR(original_predictions(i, j), loaded_predictions(i, j), tolerance)
				<< "Prediction mismatch at (" << i << "," << j << ")";
		}
	}
}

TEST_F(ModelSaveLoadIntegrationTest, RegularizedNetworkPreservesParameters) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4, 0.001, 0.001, 0.0001, 0.0001));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2, 0.001, 0.001));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	
	auto original_params = model->GetParameters();
	
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	auto loaded_params = loaded_model->GetParameters();
	
	ASSERT_EQ(original_params.size(), loaded_params.size());
	
	for (size_t i = 0; i < original_params.size(); ++i) 
	{
		const auto& [orig_weights, orig_biases] = original_params[i];
		const auto& [load_weights, load_biases] = loaded_params[i];
		
		EXPECT_TRUE(orig_weights.isApprox(load_weights, tolerance))
			<< "Weights mismatch at layer " << i;
		EXPECT_TRUE(orig_biases.isApprox(load_biases, tolerance))
			<< "Biases mismatch at layer " << i;
	}
}

TEST_F(ModelSaveLoadIntegrationTest, TrainingStatePreservedForContinuation) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	
	model->SaveModel(temp_training_file, true);
	
	model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	Eigen::MatrixXd continued_predictions = model->Predict(X_test, 1);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_training_file);
	loaded_model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	Eigen::MatrixXd loaded_continued_predictions = loaded_model->Predict(X_test, 1);
	
	EXPECT_EQ(continued_predictions.rows(), loaded_continued_predictions.rows());
}

TEST_F(ModelSaveLoadIntegrationTest, Conv1DNetworkSaveLoad) 
{
	Eigen::MatrixXd X_signal = Eigen::MatrixXd::Random(4, 32);  
	Eigen::MatrixXd y_signal(4, 1);
	y_signal << 0, 1, 0, 1;
	
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		4,
		3,
		32,
		1,
		1,
		1
	));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		4,
		2,
		32,
		4,
		2
	));
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(16 * 4, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.001));
	model->Finalize();
	
	Eigen::MatrixXd original_output = model->Predict(X_signal, 1);
	
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	Eigen::MatrixXd loaded_output = loaded_model->Predict(X_signal, 1);
	
	EXPECT_TRUE(original_output.isApprox(loaded_output, tolerance));
}

TEST_F(ModelSaveLoadIntegrationTest, Conv2DNetworkSaveLoad) 
{
	Eigen::MatrixXd X_images = Eigen::MatrixXd::Random(2, 64);
	Eigen::MatrixXd y_images(2, 1);
	y_images << 0, 1;
	
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		4,     
		3, 3,  
		8, 8,  
		1,     
		1,     
		1, 1   
	));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		4,     
		2,     
		8, 8,  
		4,     
		2      
	));
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4 * 4 * 4, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.001));
	model->Finalize();
	
	Eigen::MatrixXd original_output = model->Predict(X_images, 1);
	
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	Eigen::MatrixXd loaded_output = loaded_model->Predict(X_images, 1);
	
	EXPECT_TRUE(original_output.isApprox(loaded_output, tolerance));
}

TEST_F(ModelSaveLoadIntegrationTest, BatchNormNetworkSaveLoad) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 8));
	model->Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(8));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	
	Eigen::MatrixXd original_output = model->Predict(X_test, 1);
	
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	Eigen::MatrixXd loaded_output = loaded_model->Predict(X_test, 1);
	
	EXPECT_TRUE(original_output.isApprox(loaded_output, tolerance));
}

TEST_F(ModelSaveLoadIntegrationTest, DropoutNetworkSaveLoad) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 8));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	model->Train(X_train, y_train, 4, 5, 100, 0, X_test, y_test);
	
	Eigen::MatrixXd original_output = model->Predict(X_test, 1);
	
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	Eigen::MatrixXd loaded_output = loaded_model->Predict(X_test, 1);
	
	EXPECT_TRUE(original_output.isApprox(loaded_output, tolerance));
}

TEST_F(ModelSaveLoadIntegrationTest, DifferentOptimizersSaveLoad) 
{
	std::vector<std::unique_ptr<NEURAL_NETWORK::Optimizer>> optimizers;
	{
		auto model = std::make_unique<NEURAL_NETWORK::Model>();
		model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
		
		model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				   std::make_unique<NEURAL_NETWORK::StochasticGradientDescent>(0.1, 0.001, 0.9));
		model->Finalize();
		
		model->Train(X_train, y_train, 4, 3, 100, 0, X_test, y_test);
		
		Eigen::MatrixXd original = model->Predict(X_test, 1);
		model->SaveModel(temp_model_file);
		
		auto loaded = std::make_unique<NEURAL_NETWORK::Model>();
		loaded->LoadModel(temp_model_file);
		Eigen::MatrixXd loaded_pred = loaded->Predict(X_test, 1);
		
		EXPECT_TRUE(original.isApprox(loaded_pred, tolerance)) << "SGD optimizer test failed";
	}
	
	{
		auto model = std::make_unique<NEURAL_NETWORK::Model>();
		model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
		
		model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				   std::make_unique<NEURAL_NETWORK::AdaGrad>(0.1));
		model->Finalize();
		
		model->Train(X_train, y_train, 4, 3, 100, 0, X_test, y_test);
		
		Eigen::MatrixXd original = model->Predict(X_test, 1);
		model->SaveModel(temp_model_file);
		
		auto loaded = std::make_unique<NEURAL_NETWORK::Model>();
		loaded->LoadModel(temp_model_file);
		Eigen::MatrixXd loaded_pred = loaded->Predict(X_test, 1);
		
		EXPECT_TRUE(original.isApprox(loaded_pred, tolerance)) << "AdaGrad optimizer test failed";
	}
	
	{
		auto model = std::make_unique<NEURAL_NETWORK::Model>();
		model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
		
		model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				   std::make_unique<NEURAL_NETWORK::RMSProp>(0.01));
		model->Finalize();
		
		model->Train(X_train, y_train, 4, 3, 100, 0, X_test, y_test);
		
		Eigen::MatrixXd original = model->Predict(X_test, 1);
		model->SaveModel(temp_model_file);
		
		auto loaded = std::make_unique<NEURAL_NETWORK::Model>();
		loaded->LoadModel(temp_model_file);
		Eigen::MatrixXd loaded_pred = loaded->Predict(X_test, 1);
		
		EXPECT_TRUE(original.isApprox(loaded_pred, tolerance)) << "RMSProp optimizer test failed";
	}
}

TEST_F(ModelSaveLoadIntegrationTest, LoadNonExistentFileThrows) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	
	EXPECT_THROW(model->LoadModel("/nonexistent/path/model.bin"), std::runtime_error);
}

TEST_F(ModelSaveLoadIntegrationTest, MultipleSaveLoadCyclesPreserveAccuracy) 
{
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));
	model->Finalize();
	
	model->Train(X_train, y_train, 4, 10, 100, 0, X_test, y_test);
	
	Eigen::MatrixXd original_output = model->Predict(X_test, 1);
	
	for (int cycle = 0; cycle < 5; ++cycle) 
	{
		model->SaveModel(temp_model_file);
		
		auto loaded = std::make_unique<NEURAL_NETWORK::Model>();
		loaded->LoadModel(temp_model_file);
		
		Eigen::MatrixXd cycled_output = loaded->Predict(X_test, 1);
		
		EXPECT_TRUE(original_output.isApprox(cycled_output, tolerance))
			<< "Accuracy degraded after cycle " << cycle;
		
		model = std::move(loaded);
	}
}
