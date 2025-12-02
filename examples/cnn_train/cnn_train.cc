#include "cnn_train.h"

int cnn_train_main()
{
	std::cout << "NN_EPOCHS: " << NN_EPOCHS << std::endl;
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
	std::cout << "NN_PRINT_EVERY: " << NN_PRINT_EVERY << std::endl;
	std::string dataset_url = "https://nnfs.io/datasets/fashion_mnist_images.zip";
	std::string output_dir = "data/";
	
	Eigen::MatrixXd X, X_test, y, y_test;
	try
	{
		NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir, X, y, X_test, y_test);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load dataset: " << e.what() << std::endl;
		return 1;
	}
	
	NEURAL_NETWORK::Model model;

	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());

	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		32,
		3, 3,
		28, 28, 1,
		true,
		1, 1,
		0.0, 1e-3
	));

	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(32));

	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		64,
		2,
		28, 28, 32,
		2
	));

	model.Add(std::make_shared<NEURAL_NETWORK::Convolution2D>(
		64,
		3, 3,
		14, 14, 32,
		true,
		1, 1,
		0.0, 1e-3
	));

	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(64));

	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling>(
		64,
		2,
		14, 14, 64,
		2
	));

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5));

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		7*7*64,
		64,
		0.0, 1e-3
	));

	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(64));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.8));

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		64, 10,
		0.0, 1e-3
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.0005, 1e-7)
	);

	model.Finalize();

	model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, SAVE_EVERY, X_test, y_test);
	
	model.SaveModel("data/fashion_mnist_CNN_complete_model.bin");

	return 0;
}
