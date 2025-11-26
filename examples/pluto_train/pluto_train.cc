#include "pluto_train.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Symbol = std::pair<double, double>;

int pluto_train_main()
{
	std::cout << "NN_EPOCHS: " << NN_EPOCHS << std::endl;
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
	std::cout << "NN_PRINT_EVERY: " << NN_PRINT_EVERY << std::endl;
	Eigen::MatrixXd X_train, y_train_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("data/Pluto/rx_tx_train.csv", X_train, y_train_coords, ',');

	Eigen::MatrixXd X_test, y_test_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("data/Pluto/rx_tx_test.csv", X_test, y_test_coords, ',');

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

	model.Train(X_train, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, SAVE_EVERY, X_test, y_test);
	
	model.SaveModel("data/pluto_model_save.bin");

	std::cout << "\nEvaluating final model on test data:\n";
	model.Evaluate(X_test, y_test, BATCH_SIZE);
}