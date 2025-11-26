#include "neural_train.h"

int neural_train_main()
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

	model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, SAVE_EVERY, X_test, y_test);
	
	model.SaveModel("data/fashion_mnist_model.bin");
	
	return 0;
}