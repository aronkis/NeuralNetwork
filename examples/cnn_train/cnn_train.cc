#include "cnn_train.h"

int cnn_train_main()
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
	std::cout << "Architecture: Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.5) -> Dense(64) -> BatchNorm -> ReLU -> Dropout(0.8) -> Dense -> Softmax" << std::endl;

	// First Convolution2D Block
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		32,        // 32 filters
		3, 3,      // 3x3 filters
		28, 28, 1, // 28x28x1 input (Fashion-MNIST)
		true,      // padding
		1, 1,      // stride 1x1
		0.0, 1e-3  // DOUBLED L2 regularization again: 5e-4 to 1e-3 (10x original!)
	));

	// BatchNormalization after first Convolution2D
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

	// Second Convolution2D Block
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		64,        // 64 filters
		3, 3,      // 3x3 filters
		14, 14, 32, // 14x14x32 input from pooling
		true,      // padding
		1, 1,      // stride 1x1
		0.0, 1e-3  // DOUBLED L2 regularization again: 5e-4 to 1e-3 (10x original!)
	));

	// BatchNormalization after second Convolution2D
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

	// AGGRESSIVE dropout after final pooling layer
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5));

	// Dense Layers - AGGRESSIVELY reduced for anti-overfitting
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		7*7*64,    // Flattened size: 7×7×64 = 3136
		64,        // FURTHER reduced from 96 to 64 neurons (50% of original)
		0.0, 1e-3  // DOUBLED L2 regularization again: 5e-4 to 1e-3 (10x original!)
	));

	// Add Batch Normalization after dense layer for additional regularization
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(64));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	// MASSIVE dropout for aggressive regularization
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.8));

	// Output layer (adjusted for 64 neurons)
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		64, 10,    // 64 -> 10 classes (adjusted for further reduced hidden layer)
		0.0, 1e-3  // DOUBLED L2 regularization again: 5e-4 to 1e-3 (10x original!)
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	std::cout << "Expected data flow:" << std::endl;
	std::cout << "Input: 28x28x1 -> Conv1: 28x28x32 -> BatchNorm -> ReLU -> Pool: 14x14x32 -> Conv2: 14x14x64 -> BatchNorm -> ReLU -> Pool: 7x7x64 -> Dropout(0.5) -> Dense: 64 -> BatchNorm -> ReLU -> Dropout(0.8) -> Output: 10" << std::endl;
	
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.0005, 1e-7) // Lower learning rate for stability
	);

	model.Finalize();

	model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, SAVE_EVERY, X_test, y_test);  // AGGRESSIVE early stopping: 10 epochs max
	
	model.SaveModel("data/fashion_mnist_CNN_complete_model.bin");

	return 0;
}
