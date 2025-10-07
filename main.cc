#include "Model.h"
#include <set>

#ifndef NN_EPOCHS
#define NN_EPOCHS 100
#endif

#ifndef NN_PRINT_EVERY
#define NN_PRINT_EVERY 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 256
#endif

// #define PLUTO
#define PLUTO_MODEL
// #define FASHION_MNIST
// #define MODEL

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
			  std::make_unique<NEURAL_NETWORK::Adam>(0.001, 1e-5));

	model.Finalize();

	model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test);
	
	model.SaveModel("data/fashion_mnist_model_save_2.bin");
}
#endif