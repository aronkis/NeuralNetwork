#include "neural_eval.h"

int neural_eval_main(int argc, char **argv)
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
	try
	{
		model.LoadModel("../models/fashion_mnist_model.bin");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load model: " << e.what() << std::endl;
		return 1;
	}
	if (argc == 3)
	{
		try
		{
			NEURAL_NETWORK::Helpers::LoadData(argv[1], image, y);
		}
		catch (const std::runtime_error& e)
		{
			std::cerr << e.what() << std::endl;
			return 1;
		}
		Eigen::MatrixXd predictions = model.Predict(image, 1);
		std::cout << "Predictions for first " << argv[2] << " images:\n";
		for (int i = 0; i < std::stoi(argv[2]); i++)
		{
			std::cout << fashion_mnist_labels[static_cast<int>(predictions(i, 0))] << std::endl;
		}
	}
	else
	{
		std::string path;
		while (std::getline(std::cin, path))
		{
			if (path.empty())
				continue;
			std::string full_path = "data/extracted/test/" + path;
			try
			{
				NEURAL_NETWORK::Helpers::ReadSingleImage(full_path.c_str(), image);
			}
			catch (const std::runtime_error& e)
			{
				std::cerr << e.what() << std::endl;
				continue;
			}

			Eigen::MatrixXd predictions = model.Predict(image, 1);
			std::cout << fashion_mnist_labels[static_cast<int>(predictions(0, 0))] << std::endl;
			std::cout << "The confidence values are: " << model.GetConfidenceValues() << std::endl;
		}
	}
	
	return 0;
}