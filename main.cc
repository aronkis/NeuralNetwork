#include "Model.h"
#include <set>

#ifndef NN_EPOCHS
#define NN_EPOCHS 5
#endif

#ifndef NN_PRINT_EVERY
#define NN_PRINT_EVERY 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 128
#endif

// #define PLUTO
// #define PLUTO_MODEL
// #define FASHION_MNIST
// #define MODEL
// #define CNN
#define CNN_MODEL

#ifdef CNN

#include <iostream>
#include "Convolution.h"
#include "MaxPooling.h"
#include "BatchNormalization.h"

int main()
{
	std::cout << "NN_EPOCHS: " << NN_EPOCHS << std::endl;
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
	std::cout << "NN_PRINT_EVERY: " << NN_PRINT_EVERY << std::endl;
	std::string dataset_url = "https://nnfs.io/datasets/fashion_mnist_images.zip";
	std::string output_dir = "data/";
	
	Eigen::MatrixXd X, X_test, y, y_test;
	NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir, X, y, X_test, y_test);
	
	NEURAL_NETWORK::Model model;
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());

    // ===== COMPLETE CNN WITH MAXPOOLING + BATCH NORMALIZATION =====
    std::cout << "\n=== Testing Complete CNN with MaxPooling + BatchNormalization ===" << std::endl;
    std::cout << "Architecture: Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dense -> ReLU -> Dropout -> Dense -> Softmax" << std::endl;

    // First Convolution Block
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution>(
        32,        // 32 filters
        3, 3,      // 3x3 filters
        28, 28, 1, // 28x28x1 input (Fashion-MNIST)
        true,      // padding
        1, 1,      // stride 1x1
        0.0, 1e-4  // L2 regularization
    ));

    // BatchNormalization after first convolution
    model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(
        32         // number of channels (filters)
    ));

    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

    // MaxPooling: 28x28 -> 14x14
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        64,        // batch_size (will be adjusted dynamically)
        2,         // pool_size 2x2
        28, 28, 32, // input dimensions
        2          // stride = 2 (non-overlapping)
    ));

    // Second Convolution Block
    model.Add(std::make_shared<NEURAL_NETWORK::Convolution>(
        64,        // 64 filters
        3, 3,      // 3x3 filters
        14, 14, 32, // 14x14x32 input from pooling
        true,      // padding
        1, 1,      // stride 1x1
        0.0, 1e-4  // L2 regularization
    ));

    // BatchNormalization after second convolution
    model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(
        64         // number of channels (filters)
    ));

    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

    // MaxPooling: 14x14 -> 7x7
    model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
        64,        // batch_size (will be adjusted dynamically)
        2,         // pool_size 2x2
        14, 14, 64, // input dimensions
        2          // stride = 2 (non-overlapping)
    ));

    // Dense Layers - Transition from 7x7x64 = 3136 features
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
        7*7*64,    // Flattened size: 7×7×64 = 3136
        128,       // Hidden layer with 128 neurons
        0.0, 1e-4  // L2 regularization
    ));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

    // Dropout for regularization in dense layers
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5));

    // Output layer
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
        128, 10,   // 128 -> 10 classes
        0.0, 1e-4  // L2 regularization
    ));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

    std::cout << "Expected data flow:" << std::endl;
    std::cout << "Input: 28x28x1 -> Conv1: 28x28x32 -> BatchNorm -> ReLU -> Pool: 14x14x32 -> Conv2: 14x14x64 -> BatchNorm -> ReLU -> Pool: 7x7x64 -> Dense: 128 -> ReLU -> Dropout(0.5) -> Output: 10" << std::endl;
    
    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.0005, 1e-7) // Lower learning rate for stability
    );

    model.Finalize();

    model.Train(X, y, BATCH_SIZE, 2, NN_PRINT_EVERY, X_test, y_test);
    
    model.SaveModel("data/fashion_mnist_CNN_complete_model.bin");
}

#endif

#ifdef CNN_MODEL

#include <iostream>
#include "Convolution.h"

int main()
{
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
	std::string dataset_url = "https://nnfs.io/datasets/fashion_mnist_images.zip";
	std::string output_dir = "data/";

	Eigen::MatrixXd X, X_test, y, y_test;
	NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir, X, y, X_test, y_test);

	NEURAL_NETWORK::Model model;

	std::cout << "Loading model..." << std::endl;
	model.LoadModel("data/fashion_mnist_CNN_complete_model.bin");
	std::cout << "Model loaded successfully!" << std::endl;

	std::cout << "Testing forward pass on small batch..." << std::endl;
	// Test with just one sample first
	Eigen::MatrixXd test_sample = X_test.topRows(1);
	Eigen::MatrixXd test_pred = model.Predict(test_sample, 1);
	std::cout << "Forward pass successful!" << std::endl;

	std::cout << "Evaluating on test data..." << std::endl;
	// Evaluate the loaded model on test data
	model.Evaluate(X_test, y_test, BATCH_SIZE);

	// Example: Make predictions on a subset of test data
	int num_samples_to_predict = 10;
	Eigen::MatrixXd sample_X = X_test.topRows(num_samples_to_predict);
	Eigen::MatrixXd sample_y = y_test.topRows(num_samples_to_predict);

	Eigen::MatrixXd predictions = model.Predict(sample_X, 1);

	std::cout << "Predictions vs Actual for first " << num_samples_to_predict << " samples:" << std::endl;
	for (int i = 0; i < num_samples_to_predict; i++) {
		int predicted_class = static_cast<int>(predictions(i, 0));
		int actual_class = static_cast<int>(sample_y(i, 0));
		std::cout << "Sample " << i << ": Predicted = " << predicted_class
				  << ", Actual = " << actual_class
				  << (predicted_class == actual_class ? " ✓" : " ✗") << std::endl;
	}
}

#endif

#ifdef PLUTO_MODEL

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Symbol = std::pair<double, double>;

int main()
{
	Eigen::MatrixXd X_test;
	Eigen::MatrixXd y_test_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("../data/Pluto/rx_tx_test.csv", 
												  X_test, 
												  y_test_coords, 
												  ',');
	Eigen::MatrixXd X_test_unscaled = X_test;
	NEURAL_NETWORK::Helpers::ScaleData(X_test);

	std::set<Symbol> unique_symbols;
	for (int i = 0; i < y_test_coords.rows(); i++)
	{
		unique_symbols.insert({y_test_coords(i, 0), y_test_coords(i, 1)});
	}

	std::map<Symbol, int> symbol_to_label;
	int next_label = 0;
	for (const auto &sym : unique_symbols)
	{
		symbol_to_label[sym] = next_label++;
	}

	Eigen::MatrixXd y_test(y_test_coords.rows(), 1);
	for (int i = 0; i < y_test_coords.rows(); i++)
	{
		Symbol key{y_test_coords(i, 0), y_test_coords(i, 1)};
		y_test(i, 0) = symbol_to_label.count(key) ? symbol_to_label[key] : -1;
	}

	NEURAL_NETWORK::Model model2;
	model2.LoadModel("../data/pluto_model_save.bin");

	std::map<int, Symbol> label_to_symbol;
	for (auto const &kv : symbol_to_label)
	{
		label_to_symbol[kv.second] = kv.first;
	}

	std::cout << "\nEvaluating loaded model on test data:\n";
	model2.Evaluate(X_test, y_test, BATCH_SIZE);

	Eigen::MatrixXd y_pred_labels = model2.Predict(X_test, 1);

	std::vector<double> x_input, y_input, x_true, y_true, x_pred, y_pred;

	for (int i = 0; i < X_test_unscaled.rows(); i++)
	{
		x_input.push_back(X_test_unscaled(i, 0));
		y_input.push_back(X_test_unscaled(i, 1));
		x_true.push_back(y_test_coords(i, 0));
		y_true.push_back(y_test_coords(i, 1));
		int predicted_label = static_cast<int>(y_pred_labels(i, 0));
		if (label_to_symbol.count(predicted_label))
		{
			Symbol coords = label_to_symbol[predicted_label];
			x_pred.push_back(coords.first);
			y_pred.push_back(coords.second);
		}
	}

	plt::figure_size(1200, 1000);
	plt::named_plot("Input", x_input, y_input, "o");
	plt::named_plot("True Values", x_true, y_true, "o");
	plt::named_plot("Predictions", x_pred, y_pred, "o");
	plt::title("Symbol Prediction");
	plt::xlabel("I");
	plt::ylabel("Q");
	plt::legend();
	plt::grid(true);
	plt::show();
}
#endif

#ifdef PLUTO

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Symbol = std::pair<double, double>;

int main()
{
	std::cout << "NN_EPOCHS: " << NN_EPOCHS << std::endl;
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
	std::cout << "NN_PRINT_EVERY: " << NN_PRINT_EVERY << std::endl;
	Eigen::MatrixXd X_train, y_train_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("../data/Pluto/rx_tx_train.csv", X_train, y_train_coords, ',');

	Eigen::MatrixXd X_test, y_test_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("../data/Pluto/rx_tx_test.csv", X_test, y_test_coords, ',');

	Eigen::MatrixXd X_test_unscaled = X_test;
	NEURAL_NETWORK::Helpers::ScaleData(X_train);
	NEURAL_NETWORK::Helpers::ScaleData(X_test);

	std::set<Symbol> unique_symbols;
	for (int i = 0; i < y_train_coords.rows(); i++)
	{
		unique_symbols.insert({y_train_coords(i, 0), y_train_coords(i, 1)});
	}

	std::map<Symbol, int> symbol_to_label;
	int next_label = 0;
	for (auto const &sym : unique_symbols)
	{
		symbol_to_label[sym] = next_label++;
	}

	const int n_classes = next_label;
	std::cout << "Number of unique symbols (classes): " << n_classes << std::endl;

	Eigen::MatrixXd y_train(y_train_coords.rows(), 1);
	for (int i = 0; i < y_train_coords.rows(); i++)
	{
		y_train(i, 0) = symbol_to_label[{y_train_coords(i, 0), y_train_coords(i, 1)}];
	}

	Eigen::MatrixXd y_test(y_test_coords.rows(), 1);
	for (int i = 0; i < y_test_coords.rows(); i++)
	{
		Symbol key{y_test_coords(i, 0), y_test_coords(i, 1)};
		y_test(i, 0) = symbol_to_label.count(key) ? symbol_to_label[key] : -1;
	}
	NEURAL_NETWORK::Helpers::ShuffleData(X_train, y_train);

	NEURAL_NETWORK::Model model;
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(X_train.cols(), 1024, 0.0, 1e-5, 0.0, 1e-5));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(1024, 1024, 0.0, 1e-5, 0.0, 1e-5));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(1024, 512, 0.0, 1e-5, 0.0, 1e-5));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(512, n_classes, 0.0, 1e-5, 0.0, 1e-5));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	model.Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
			  std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(), 
			  std::make_unique<NEURAL_NETWORK::Adam>(0.0002, 5e-5));

	model.Finalize();

	model.Train(X_train, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test);
	
	model.SaveModel("../data/pluto_model_save.bin");

	std::cout << "\nEvaluating final model on test data:\n";
	model.Evaluate(X_test, y_test, BATCH_SIZE);
}
#endif

#ifdef MODEL
#include <iostream>
#include <map>
int main(int argc, char **argv)
{
	std::map<int, std::string> fashion_mnist_labels = {{0, "T-shirt/top"},
													   {1, "Trouser"}, 
													   {2, "Pullover"}, 
													   {3, "Dress"}, 
													   {4, "Coat"},
													   {5, "Sandal"}, 
													   {6, "Shirt"}, 
													   {7, "Sneaker"}, 
													   {8, "Bag"}, 
													   {9, "Ankle boot"}};
	Eigen::MatrixXd image, y;
	NEURAL_NETWORK::Model model;
	model.LoadModel("data/fashion_mnist_model_save_2.bin");
	if (argc == 3)
	{
		NEURAL_NETWORK::Helpers::LoadData(argv[1], image, y);
		Eigen::MatrixXd predictions = model.Predict(image, 1);
		std::cout << "Predictions for first " << argv[2] << " images:\n";
		for (int i = 0; i < std::stoi(argv[2]); i++)
			std::cout << fashion_mnist_labels[static_cast<int>(predictions(i, 0))] << std::endl;
	}
	else
	{
		std::string path;
		while (std::getline(std::cin, path))
		{
			if (path.empty())
				continue;
			std::string full_path = "data/extracted/test/" + path;
			NEURAL_NETWORK::Helpers::ReadSingleImage(full_path.c_str(), image);
			Eigen::MatrixXd predictions = model.Predict(image, 1);
			std::cout << fashion_mnist_labels[static_cast<int>(predictions(0, 0))] << std::endl;
		}
	}
	return 0;
}
#endif

#ifdef FASHION_MNIST
int main()
{
	std::string dataset_url = "https://nnfs.io/datasets/fashion_mnist_images.zip";
	std::string output_dir = "data/";
	
	Eigen::MatrixXd X, X_test, y, y_test;
	NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir, X, y, X_test, y_test);
	
	NEURAL_NETWORK::Model model;
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(X.cols(), 128, 0.0, 5e-4, 0.0, 5e-4));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(128, 128));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(128, 10));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());
	
	model.Set(std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(), 
			  std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(), 
			  std::make_unique<NEURAL_NETWORK::Adam>(0.0001, 1e-7));

	model.Finalize();

	model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test);
	
	model.SaveModel("data/fashion_mnist_model_save_2.bin");
}
#endif


/* 
CNN
========== Epoch 10/10 ==========
Step: 0 [1.06%], Accuracy: 0.953125, Loss: 0.205157, LR: 0.000499958
Step: 12 [13.83%], Accuracy: 0.96875, Loss: 0.142752, LR: 0.000499957
Step: 24 [26.60%], Accuracy: 0.9375, Loss: 0.142313, LR: 0.000499957
Step: 36 [39.36%], Accuracy: 0.984375, Loss: 0.118973, LR: 0.000499956
Step: 48 [52.13%], Accuracy: 0.953125, Loss: 0.0935616, LR: 0.000499955
Step: 60 [64.89%], Accuracy: 0.9375, Loss: 0.202393, LR: 0.000499955
Step: 72 [77.66%], Accuracy: 0.96875, Loss: 0.1327, LR: 0.000499954
Step: 84 [90.43%], Accuracy: 0.953125, Loss: 0.14065, LR: 0.000499954
Step: 93 [100.00%], Accuracy: 0.979167, Loss: 0.147982, LR: 0.000499953
Training: Accuracy: 0.937833, Loss: 0.194663 (Data loss: 0.194663 | Regularization loss: 0), Learning Rate: 0.000499953
Validation Accuracy: 0.835, Validation Loss: 0.59472

CNN -> MaxPool
========== Epoch 10/10 ==========
Step: 0 [1.06%], Accuracy: 0.96875, Loss: 0.217884, LR: 0.000499958
Step: 12 [13.83%], Accuracy: 0.953125, Loss: 0.280522, LR: 0.000499957
Step: 24 [26.60%], Accuracy: 0.90625, Loss: 0.468903, LR: 0.000499957
Step: 36 [39.36%], Accuracy: 0.84375, Loss: 0.358572, LR: 0.000499956
Step: 48 [52.13%], Accuracy: 0.875, Loss: 0.432397, LR: 0.000499955
Step: 60 [64.89%], Accuracy: 0.921875, Loss: 0.255831, LR: 0.000499955
Step: 72 [77.66%], Accuracy: 0.984375, Loss: 0.182179, LR: 0.000499954
Step: 84 [90.43%], Accuracy: 0.984375, Loss: 0.140204, LR: 0.000499954
Step: 93 [100.00%], Accuracy: 0.895833, Loss: 0.320941, LR: 0.000499953
Training: Accuracy: 0.921333, Loss: 0.280477 (Data loss: 0.233282 | Regularization loss: 0.0471946), Learning Rate: 0.000499953
Validation Accuracy: 0.84, Validation Loss: 0.441634

CNN -> MaxPool -> BatchNorm -> Dense -> Dropout
========== Epoch 10/10 ==========
Step: 0 [1.06%], Accuracy: 0.796875, Loss: 0.517113, LR: 0.000499958
Step: 12 [13.83%], Accuracy: 0.859375, Loss: 0.412646, LR: 0.000499957
Step: 24 [26.60%], Accuracy: 0.78125, Loss: 0.575299, LR: 0.000499957
Step: 36 [39.36%], Accuracy: 0.765625, Loss: 0.662682, LR: 0.000499956
Step: 48 [52.13%], Accuracy: 0.8125, Loss: 0.470909, LR: 0.000499955
Step: 60 [64.89%], Accuracy: 0.90625, Loss: 0.361092, LR: 0.000499955
Step: 72 [77.66%], Accuracy: 0.828125, Loss: 0.442175, LR: 0.000499954
Step: 84 [90.43%], Accuracy: 0.828125, Loss: 0.532051, LR: 0.000499954
Step: 93 [100.00%], Accuracy: 0.958333, Loss: 0.253314, LR: 0.000499953
Training: Accuracy: 0.853167, Loss: 0.456743 (Data loss: 0.408862 | Regularization loss: 0.0478814), Learning Rate: 0.000499953
Validation Accuracy: 0.863, Validation Loss: 0.417673


CNN -> MaxPool -> BatchNorm(trained) -> Dense -> Dropout (FULL SET)
========== Epoch 5/5 ==========
Step: 0 [0.21%], Accuracy: 0.84375, Loss: 0.466412, LR: 0.000499906
Step: 50 [10.87%], Accuracy: 0.835938, Loss: 0.424009, LR: 0.000499904
Step: 100 [21.54%], Accuracy: 0.875, Loss: 0.391405, LR: 0.000499901
Step: 150 [32.20%], Accuracy: 0.835938, Loss: 0.478866, LR: 0.000499899
Step: 200 [42.86%], Accuracy: 0.898438, Loss: 0.279198, LR: 0.000499896
Step: 250 [53.52%], Accuracy: 0.851562, Loss: 0.49053, LR: 0.000499894
Step: 300 [64.18%], Accuracy: 0.851562, Loss: 0.420675, LR: 0.000499891
Step: 350 [74.84%], Accuracy: 0.851562, Loss: 0.404609, LR: 0.000499889
Step: 400 [85.50%], Accuracy: 0.882812, Loss: 0.397719, LR: 0.000499886
Step: 450 [96.16%], Accuracy: 0.898438, Loss: 0.432853, LR: 0.000499884
Step: 468 [100.00%], Accuracy: 0.8125, Loss: 0.54317, LR: 0.000499883
Training: Accuracy: 0.869617, Loss: 0.414934 (Data loss: 0.366657 | Regularization loss: 0.0482769), Learning Rate: 0.000499883
Validation Accuracy: 0.8887, Validation Loss: 0.308375
Model saved to "data/fashion_mnist_CNN_complete_model.bin"

*/

/*
TODO:
Memory Management

  - Reduce memory allocations in training loops: In Model::Train(), consider pre-allocating
  matrices for weight_update, bias_update etc. outside the training loop to avoid repeated
  allocations
  - Eigen memory alignment: Add EIGEN_MAKE_ALIGNED_OPERATOR_NEW macros to classes with Eigen
  members for better vectorization
  - Tensor operations: In Convolution.cc, the tensor-to-matrix conversions could be optimized with
   better memory layouts

  Computational Efficiency

  - BLAS optimization: The comment about EIGEN_USE_BLAS in Convolution.h suggests this could be
  enabled for better linear algebra performance
  - Parallelize batch operations: The batch processing in Model::Train() could benefit from OpenMP
   parallelization
  - Vectorized operations: Some loops in TensorUtils.cc could be replaced with Eigen's vectorized
  operations

  🏗️ Architecture & Design Improvements

  Layer System

  - Abstract Factory Pattern: Create a LayerFactory for dynamic layer creation from config, making
   model loading more extensible
  - Layer Registration System: Implement a registry pattern so new layers can self-register for
  serialization
  - Visitor Pattern: For operations like parameter counting, gradient clipping, etc.

  Model Configuration

  - YAML/JSON Config: Replace the hard-coded model architecture in main.cc with external
  configuration files
  - Builder Pattern: Implement a ModelBuilder class for more flexible model construction
  - Hyperparameter Management: Create a dedicated hyperparameter class with validation

  🛡️ Error Handling & Robustness

  Input Validation

  - Dimension Checking: Add comprehensive dimension validation in all layer forward/backward
  methods
  - Data Validation: Validate input ranges, check for NaN/infinity values
  - Configuration Validation: Validate optimizer parameters, layer parameters at construction time

  Exception Safety

  - RAII: Ensure all resource management follows RAII principles
  - Custom Exceptions: Replace std::cerr error messages with proper exception handling
  - Rollback Mechanisms: For failed model loads/saves

  🔧 Code Quality Improvements

  Modern C++ Features

  - Smart Pointers: Consistent use of std::unique_ptr/std::shared_ptr (already partially done)
  - Move Semantics: Add move constructors/assignment operators for heavy classes
  - Constexpr: Make compile-time constants constexpr where possible
  - Auto: Use auto more consistently for type deduction

  Interface Design

  - Pure Virtual Interface: Create IOptimizer, ILoss, IAccuracy interfaces for better testability
  - Const Correctness: Many getter methods could return const& instead of copies
  - Method Chaining: Allow fluent interface for model building (model.Add().Add().Set())

  📊 Numerical Stability

  Gradient Handling

  - Gradient Clipping: Add global gradient clipping to prevent exploding gradients
  - Gradient Accumulation: For very large models that don't fit in memory
  - Mixed Precision: Consider float16 support for memory efficiency

  Numerical Precision

  - Epsilon Handling: Centralize epsilon values and make them configurable
  - Overflow/Underflow Protection: Add checks in activation functions and loss calculations
  - Numerical Derivatives: Add gradient checking for debugging

  🧪 Testing & Validation

  Unit Testing

  - Layer Testing: Individual tests for each layer's forward/backward pass
  - Gradient Tests: Numerical gradient checking
  - Serialization Tests: Round-trip testing for model save/load

  Integration Testing

  - End-to-End Tests: Complete training pipelines with known datasets
  - Performance Benchmarks: Track training speed and memory usage
  - Convergence Tests: Verify models converge on simple synthetic datasets

  📝 Documentation & Maintainability

  Code Documentation

  - Doxygen Comments: Add comprehensive API documentation
  - Mathematical Formulas: Document the math behind each layer
  - Usage Examples: More comprehensive examples beyond Fashion-MNIST

  Logging & Monitoring

  - Structured Logging: Replace std::cout with a proper logging framework
  - Training Metrics: More detailed metrics tracking (gradients, activations, etc.)
  - Progress Visualization: Better progress reporting during training

  🎯 Specific Technical Suggestions

  BatchNormalization

  - Running Statistics: The momentum update could be made more numerically stable
  - Training/Inference Mode: Make the mode switching more explicit

  Convolution Layer

  - Memory Layout: Consider NCHW vs NHWC layout optimization based on hardware
  - Winograd Convolution: For 3x3 convolutions, Winograd algorithm could be faster
  - Im2Col Optimization: The current implementation could be vectorized better

  Model Serialization

  - Version Compatibility: Add version numbers to saved models
  - Incremental Loading: For very large models, support streaming/partial loading
  - Compression: Add optional compression for saved models

  Data Loading

  - Async Loading: In Helpers.cc, implement asynchronous data loading
  - Data Augmentation: Built-in support for common augmentations
  - Memory Mapping: For very large datasets

  🔄 Long-term Architectural Considerations

  Modularity

  - Plugin System: Allow loading custom layers/optimizers as plugins
  - Backend Abstraction: Prepare for GPU/other accelerator support
  - Distributed Training: Design for future distributed training support

  Performance Monitoring

  - Profiling Integration: Built-in profiling hooks
  - Memory Tracking: Track peak memory usage
  - Computational Graph: For advanced optimizations

  These suggestions range from simple code quality improvements to more significant architectural
  changes. The priority would depend on your current needs - performance, maintainability, or
  feature completeness.
*/