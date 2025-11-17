#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>
#include <filesystem>
#include <fstream>
#include "Model.h"
#include "LayerInput.h"
#include "LayerDense.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"
#include "AccuracyCategorical.h"
#include "Adam.h"

namespace
{
	std::string WriteMinimalModelFile(const std::string& path)
	{
		std::ofstream of(path, std::ios::binary);
		if (!of)
		{
			return path;
		}

		size_t zero_size_t = 0;
		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));
		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));
		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));
		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));
		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		of.write(reinterpret_cast<const char*>(&zero_size_t), sizeof(zero_size_t));

		bool softmax_flag = false;
		of.write(reinterpret_cast<const char*>(&softmax_flag), sizeof(softmax_flag));

		return path;
	}
}

class ModelTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		model = std::make_unique<NEURAL_NETWORK::Model>();

		X_train = Eigen::MatrixXd(4, 2);
		X_train << 1.0, 2.0,
				2.0, 3.0,
				3.0, 4.0,
				4.0, 5.0;

		y_train = Eigen::MatrixXd(4, 1);
		y_train << 0, 1, 0, 1;  

		X_test = Eigen::MatrixXd(2, 2);
		X_test << 1.5, 2.5,
				3.5, 4.5;

		y_test = Eigen::MatrixXd(2, 1);
		y_test << 0, 1;

		y_train_multiclass = Eigen::MatrixXd(4, 1);
		y_train_multiclass << 0, 1, 0, 1;  

		y_test_multiclass = Eigen::MatrixXd(2, 1);
		y_test_multiclass << 0, 1;

		temp_model_file = "/tmp/test_model.bin";
	}

	void TearDown() override 
	{
		if (std::filesystem::exists(temp_model_file)) 
		{
			std::filesystem::remove(temp_model_file);
		}
	}

	void BuildSimpleClassificationModel() 
	{
		model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));  
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
		model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));  
		model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

		model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				   std::make_unique<NEURAL_NETWORK::Adam>(0.01));

		model->Finalize();
	}

	std::unique_ptr<NEURAL_NETWORK::Model> model;
	Eigen::MatrixXd X_train, y_train, X_test, y_test;
	Eigen::MatrixXd y_train_multiclass, y_test_multiclass;
	std::string temp_model_file;
	const double tolerance = 1e-6;
};

TEST_F(ModelTest, EmptyModelCanBeCreated) 
{
	EXPECT_NO_THROW(NEURAL_NETWORK::Model test_model);
}

TEST_F(ModelTest, LayersCanBeAdded) 
{
	auto input_layer = std::make_shared<NEURAL_NETWORK::LayerInput>();
	auto dense_layer = std::make_shared<NEURAL_NETWORK::LayerDense>(2, 3);

	EXPECT_NO_THROW(model->Add(input_layer));
	EXPECT_NO_THROW(model->Add(dense_layer));
}

TEST_F(ModelTest, ModelComponentsCanBeSet) 
{
	auto loss = std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>();
	auto accuracy = std::make_unique<NEURAL_NETWORK::AccuracyCategorical>();
	auto optimizer = std::make_unique<NEURAL_NETWORK::Adam>();

	EXPECT_NO_THROW(model->Set(std::move(loss), std::move(accuracy), std::move(optimizer)));
}

TEST_F(ModelTest, ModelCanBeFinalized) 
{
	BuildSimpleClassificationModel();
	Eigen::MatrixXd predictions = model->Predict(X_train, 2);
	EXPECT_EQ(predictions.rows(), X_train.rows());
	EXPECT_EQ(predictions.cols(), 1);
}

TEST_F(ModelTest, ModelCanTrain) 
{
	BuildSimpleClassificationModel();

	auto before = model->GetParameters();

	model->Train(X_train, y_train_multiclass, 2, 2, 1, X_test, y_test_multiclass);

	auto after = model->GetParameters();
	bool changed = false;
	for (size_t i = 0; i < before.size(); i++)
	{
		if (!after[i].first.isApprox(before[i].first) ||
			!after[i].second.isApprox(before[i].second))
		{
			changed = true;
			break;
		}
	}
	EXPECT_TRUE(changed);
}

TEST_F(ModelTest, TrainingWithDifferentBatchSizes) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 1, 1, 1, X_test, y_test_multiclass);
	Eigen::MatrixXd preds_batch1 = model->Predict(X_test, 1);
	EXPECT_EQ(preds_batch1.rows(), X_test.rows());
	EXPECT_EQ(preds_batch1.cols(), 1);

	model = std::make_unique<NEURAL_NETWORK::Model>();
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 4, 1, 1, X_test, y_test_multiclass);
	Eigen::MatrixXd preds_batch4 = model->Predict(X_test, 1);
	EXPECT_EQ(preds_batch4.rows(), X_test.rows());
	EXPECT_EQ(preds_batch4.cols(), 1);
}

TEST_F(ModelTest, TrainingWithLargeBatchSize) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 10, 1, 1, X_test, y_test_multiclass);
	Eigen::MatrixXd preds = model->Predict(X_test, 1);
	EXPECT_EQ(preds.rows(), X_test.rows());
	EXPECT_EQ(preds.cols(), 1);
}

TEST_F(ModelTest, ModelCanPredict) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	Eigen::MatrixXd predictions = model->Predict(X_test, 1);

	EXPECT_EQ(predictions.rows(), X_test.rows());
	EXPECT_EQ(predictions.cols(), 1);  

	for (int i = 0; i < predictions.rows(); i++) 
	{
		int pred_class = static_cast<int>(predictions(i, 0));
		EXPECT_GE(pred_class, 0);
		EXPECT_LT(pred_class, 3);  
	}
}

TEST_F(ModelTest, PredictionWithDifferentBatchSizes) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	Eigen::MatrixXd pred1 = model->Predict(X_test, 1);
	Eigen::MatrixXd pred2 = model->Predict(X_test, 2);

	EXPECT_TRUE(pred1.isApprox(pred2, tolerance));
}

TEST_F(ModelTest, PredictionOnSingleSample) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	Eigen::MatrixXd single_sample = X_test.row(0);
	Eigen::MatrixXd single_pred = model->Predict(single_sample, 1);

	EXPECT_EQ(single_pred.rows(), 1);
	EXPECT_EQ(single_pred.cols(), 1);
}

TEST_F(ModelTest, ModelCanEvaluate) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	EXPECT_NO_THROW(model->Evaluate(X_test, y_test_multiclass, 1));
}

TEST_F(ModelTest, EvaluationWithDifferentBatchSizes) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	EXPECT_NO_THROW(model->Evaluate(X_test, y_test_multiclass, 1));
	EXPECT_NO_THROW(model->Evaluate(X_test, y_test_multiclass, 2));
}

TEST_F(ModelTest, ParametersCanBeRetrieved) 
{
	BuildSimpleClassificationModel();

	auto parameters = model->GetParameters();

	EXPECT_GT(parameters.size(), 0);

	for (const auto& param_pair : parameters) 
	{
		const auto& weights = param_pair.first;
		const auto& biases = param_pair.second;

		EXPECT_GT(weights.rows(), 0);
		EXPECT_GT(weights.cols(), 0);
		EXPECT_GT(biases.cols(), 0);
		EXPECT_EQ(biases.rows(), 1);
	}
}

TEST_F(ModelTest, ParametersCanBeSetAndRetrieved) 
{
	BuildSimpleClassificationModel();

	auto initial_params = model->GetParameters();

	auto modified_params = initial_params;
	for (auto& param_pair : modified_params) 
	{
		param_pair.first.array() += 1.0;
		param_pair.second.array() += 1.0;
	}

	model->SetParameters(modified_params);

	auto retrieved_params = model->GetParameters();

	EXPECT_EQ(retrieved_params.size(), modified_params.size());

	for (size_t i = 0; i < retrieved_params.size(); i++) 
	{
		EXPECT_TRUE(retrieved_params[i].first.isApprox(modified_params[i].first, tolerance));
		EXPECT_TRUE(retrieved_params[i].second.isApprox(modified_params[i].second, tolerance));
	}
}

TEST_F(ModelTest, ModelCanBeSaved) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	EXPECT_NO_THROW(model->SaveModel(temp_model_file));

	EXPECT_TRUE(std::filesystem::exists(temp_model_file));
}

TEST_F(ModelTest, ModelCanBeLoaded) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 2, 1, X_test, y_test_multiclass);
	model->SaveModel(temp_model_file);

	Eigen::MatrixXd original_predictions = model->Predict(X_test, 1);

	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	EXPECT_NO_THROW(loaded_model->LoadModel(temp_model_file));

	Eigen::MatrixXd loaded_predictions = loaded_model->Predict(X_test, 1);

	EXPECT_TRUE(original_predictions.isApprox(loaded_predictions, tolerance));
}

TEST_F(ModelTest, SaveParametersOnly) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);

	std::string param_file = "/tmp/test_params.bin";

	EXPECT_NO_THROW(model->SaveParameters(param_file));

	EXPECT_TRUE(std::filesystem::exists(param_file));

	if (std::filesystem::exists(param_file)) 
	{
		std::filesystem::remove(param_file);
	}
}

TEST_F(ModelTest, LoadParametersOnly) 
{
	BuildSimpleClassificationModel();

	model->Train(X_train, y_train_multiclass, 2, 2, 1, X_test, y_test_multiclass);

	std::string param_file = "/tmp/test_params.bin";
	model->SaveParameters(param_file);

	auto original_params = model->GetParameters();

	auto new_model = std::make_unique<NEURAL_NETWORK::Model>();
	new_model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	new_model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 4));
	new_model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	new_model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
	new_model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	new_model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
				   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
				   std::make_unique<NEURAL_NETWORK::Adam>(0.01));

	new_model->Finalize();

	EXPECT_NO_THROW(new_model->LoadParameters(param_file));

	auto loaded_params = new_model->GetParameters();
	EXPECT_EQ(loaded_params.size(), original_params.size());

	for (size_t i = 0; i < loaded_params.size(); i++) 
	{
		EXPECT_TRUE(loaded_params[i].first.isApprox(original_params[i].first, tolerance));
		EXPECT_TRUE(loaded_params[i].second.isApprox(original_params[i].second, tolerance));
	}

	if (std::filesystem::exists(param_file)) 
	{
		std::filesystem::remove(param_file);
	}
}

TEST_F(ModelTest, ModelWithDifferentArchitectures) 
{
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 8));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(8, 4));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(4, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));

	EXPECT_NO_THROW(model->Finalize());
	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);
	Eigen::MatrixXd preds = model->Predict(X_test, 1);
	EXPECT_EQ(preds.rows(), X_test.rows());
	EXPECT_EQ(preds.cols(), 1);
}

TEST_F(ModelTest, SingleLayerModel) 
{
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 2));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	model->Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			   std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
			   std::make_unique<NEURAL_NETWORK::Adam>(0.01));

	EXPECT_NO_THROW(model->Finalize());
	model->Train(X_train, y_train_multiclass, 2, 1, 1, X_test, y_test_multiclass);
	Eigen::MatrixXd preds = model->Predict(X_test, 1);
	EXPECT_EQ(preds.rows(), X_test.rows());
	EXPECT_EQ(preds.cols(), 1);
}

TEST_F(ModelTest, EmptyDataHandling) 
{
	BuildSimpleClassificationModel();

	Eigen::MatrixXd empty_X(0, 2);
	Eigen::MatrixXd empty_y(0, 1);
	auto before = model->GetParameters();

	EXPECT_NO_THROW(model->Train(empty_X, empty_y, 1, 1, 1, empty_X, empty_y));

	auto after = model->GetParameters();
	for (size_t i = 0; i < before.size(); i++)
	{
		EXPECT_TRUE(after[i].first.isApprox(before[i].first));
		EXPECT_TRUE(after[i].second.isApprox(before[i].second));
	}
}

TEST_F(ModelTest, LoadModelGracefullyHandlesCorruptedFile)
{
	std::string corrupt_path = WriteMinimalModelFile("/tmp/corrupt_model.bin");

	EXPECT_NO_THROW(model->LoadModel(corrupt_path));
	EXPECT_TRUE(model->GetParameters().empty());

	if (std::filesystem::exists(corrupt_path))
	{
		std::filesystem::remove(corrupt_path);
	}

	BuildSimpleClassificationModel();
	EXPECT_NO_THROW(model->Train(X_train, y_train_multiclass, 1, 1, 1, X_test, y_test_multiclass));
}

TEST_F(ModelTest, SingleSampleTraining) 
{
	BuildSimpleClassificationModel();

	Eigen::MatrixXd single_X = X_train.row(0);
	Eigen::MatrixXd single_y = y_train_multiclass.row(0);

	auto before = model->GetParameters();

	model->Train(single_X, single_y, 1, 1, 1, single_X, single_y);

	auto after = model->GetParameters();
	bool changed = false;
	for (size_t i = 0; i < before.size(); i++)
	{
		if (!after[i].first.isApprox(before[i].first) ||
			!after[i].second.isApprox(before[i].second))
		{
			changed = true;
			break;
		}
	}
	EXPECT_TRUE(changed);
}

// Tests for MaxPooling layer serialization (new functionality)
TEST_F(ModelTest, SaveLoadModelWithMaxPooling)
{
	std::string temp_model_file = "test_maxpool_model.bin";
	
	// Build CNN model with MaxPooling
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	
	// Add Convolution2D layer
	model->Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		8, 3, 3, 28, 28, 1, true, 1, 1, 0.0, 1e-4));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	
	// Add MaxPooling layer
	model->Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		32, 2, 28, 28, 8, 2));
	
	// Add Dense layer for classification
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(14*14*8, 10));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>());
	
	model->Finalize();
	
	// Save the model
	EXPECT_NO_THROW(model->SaveModel(temp_model_file));
	EXPECT_TRUE(std::filesystem::exists(temp_model_file));
	
	// Load the model
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	EXPECT_NO_THROW(loaded_model->LoadModel(temp_model_file));
	
	// Test that loaded model can make predictions
	Eigen::MatrixXd test_input = Eigen::MatrixXd::Random(2, 28*28);
	EXPECT_NO_THROW(loaded_model->Predict(test_input, 2));
	
	// Cleanup
	if (std::filesystem::exists(temp_model_file)) {
		std::filesystem::remove(temp_model_file);
	}
}

TEST_F(ModelTest, MaxPoolingParametersSavedCorrectly)
{
	std::string temp_model_file = "test_maxpool_params.bin";
	
	// Create model with specific MaxPooling parameters
	auto model = std::make_unique<NEURAL_NETWORK::Model>();
	model->Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model->Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		4, 5, 5, 16, 16, 1, false, 2, 2)); // stride 2 conv
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	
	// MaxPooling with specific parameters
	auto maxpool = std::make_shared<NEURAL_NETWORK::MaxPooling>(
		16, 3, 6, 6, 4, 3); // pool_size=3, stride=3, input 6x6x4
	model->Add(maxpool);
	
	model->Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2*2*4, 5));
	model->Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model->Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>());
	
	model->Finalize();
	
	// Test forward pass before saving
	Eigen::MatrixXd test_input = Eigen::MatrixXd::Random(1, 16*16);
	Eigen::MatrixXd original_output = model->Predict(test_input, 1);
	
	// Save and load
	model->SaveModel(temp_model_file);
	
	auto loaded_model = std::make_unique<NEURAL_NETWORK::Model>();
	loaded_model->LoadModel(temp_model_file);
	
	// Test that loaded model produces same output
	Eigen::MatrixXd loaded_output = loaded_model->Predict(test_input, 1);
	
	// Outputs should be very close (allowing for small numerical differences)
	const double tolerance = 1e-10;
	for (int i = 0; i < original_output.rows(); ++i) {
		for (int j = 0; j < original_output.cols(); ++j) {
			EXPECT_NEAR(original_output(i, j), loaded_output(i, j), tolerance)
				<< "Mismatch at (" << i << "," << j << ")";
		}
	}
	
	// Cleanup
	if (std::filesystem::exists(temp_model_file)) {
		std::filesystem::remove(temp_model_file);
	}
}
