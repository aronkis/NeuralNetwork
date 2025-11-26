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
	model.LoadModel("data/fashion_mnist_model.bin");
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