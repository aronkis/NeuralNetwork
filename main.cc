#include "Model.h"
#include <iostream>
#include <map>

#ifndef NN_EPOCHS
#define NN_EPOCHS 5
#endif

#ifndef NN_PRINT_EVERY
#define NN_PRINT_EVERY 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 256
#endif

// #define CLASSIFICATION
// #define BINARY_CLASSIFICATION
// #define REGRESSION
#define FASHION_MNIST
// #define MODEL

#ifdef MODEL
int main(int argc, char** argv)
{
	std::map<int, std::string> fashion_mnist_labels = {
		{0, "T-shirt/top"},
		{1, "Trouser"},
		{2, "Pullover"},
		{3, "Dress"},
		{4, "Coat"},
		{5, "Sandal"},
		{6, "Shirt"},
		{7, "Sneaker"},
		{8, "Bag"},
		{9, "Ankle boot"}
	};

	Eigen::MatrixXd image;
	Eigen::MatrixXd y;
	
	NEURAL_NETWORK::Model model;
	model.LoadModel("data/fashion_mnist_model_save_2.bin");
	
	if (argc == 3)
	{
		NEURAL_NETWORK::Helpers::LoadData(argv[1], image, y);
		Eigen::MatrixXd predictions = model.Predict(image, 1);
		std::cout << "Predictions for first " << argv[2] << " images:\n";
		for (int i = 0; i < std::stoi(argv[2]); ++i) 
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
			{
				continue;
			} 
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

    Eigen::MatrixXd X;
    Eigen::MatrixXd X_test;
    Eigen::MatrixXd y;
    Eigen::MatrixXd y_test;

    NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url,
                                            output_dir,
                                            X, y,
                                            X_test, y_test);
	
   	NEURAL_NETWORK::Model model;

    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(X.cols(), 128, 0.0, 5e-4, 0.0, 5e-4));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(128, 128)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(128, 10)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.001, 1e-5)
    );

    model.Finalize();

    model.Train(X, y, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test, false);
	model.SaveModel("data/fashion_mnist_model_save_2.bin");
}
#endif

#ifdef REGRESSION
int main()
{
    NEURAL_NETWORK::Model model;

    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(1, 64, 0.0, 5e-4, 0.0, 5e-4));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 64)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationLinear>());

    model.Set(
        std::make_unique<NEURAL_NETWORK::LossMeanSquaredError>(),
        std::make_unique<NEURAL_NETWORK::AccuracyRegression>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.005, 1e-3, 0.9, 0.999, 1e-7)
    );

    model.Finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "data/sine.txt";
    std::string validation_filename = "data/sine_validation.txt";
    NEURAL_NETWORK::Helpers::Read1DIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::Read1DIntoEigen(validation_filename, X_test, y_test);

    model.Train(X_train, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test, true);

}
#endif

#ifdef BINARY_CLASSIFICATION
int main()
{
	NEURAL_NETWORK::Model model;

    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 64, 0, 5e-4, 0, 5e-4));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSigmoid>());

    model.Set(
        std::make_unique<NEURAL_NETWORK::LossBinaryCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.019, 5e-7, 0.9, 0.999, 1e-7)
    );

    model.Finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "data/binary.txt";
    std::string validation_filename = "data/binary_validation.txt";
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

    model.Train(X_train, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test, false);

	std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> params = model.GetParameters();

	model.SaveParameters("data/binary_model.bin");

	model.SaveModel("data/binary_model_save.bin");

	NEURAL_NETWORK::Model model2;

    model2.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model2.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 64, 0, 5e-4, 0, 5e-4));
    model2.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model2.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model2.Add(std::make_shared<NEURAL_NETWORK::ActivationSigmoid>());

    model2.Set(
        std::make_unique<NEURAL_NETWORK::LossBinaryCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>()
    );

    model2.Finalize();

	model2.SetParameters(params);

	NEURAL_NETWORK::Model model3;

    model3.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model3.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 64, 0, 5e-4, 0, 5e-4));
    model3.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model3.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model3.Add(std::make_shared<NEURAL_NETWORK::ActivationSigmoid>());

    model3.Set(
        std::make_unique<NEURAL_NETWORK::LossBinaryCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>()
    );

    model3.Finalize();
	model3.LoadParameters("data/binary_model.bin");



	std::cout << "Evaluation model 1: " << std::endl;
	model.Evaluate(X_test, y_test, 1);

	std::cout << "Evaluation model 2: " << std::endl;
	model2.Evaluate(X_test, y_test, 1);

	std::cout << "Evaluation model 3: " << std::endl;
	model3.Evaluate(X_test, y_test, 1);

	NEURAL_NETWORK::Model model4;
	model4.LoadModel("data/binary_model_save.bin");
	std::cout << "Evaluation model 4: " << std::endl;
	model4.Evaluate(X_test, y_test, 1);

}
#endif

#ifdef CLASSIFICATION
int main() {
    NEURAL_NETWORK::Model model;

    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(2, 512, 0.0, 5e-4, 0.0, 5e-4));
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.1));
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(512, 3)); 
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy>());

    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.05, 5e-5)
    );

    model.Finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "data/points.txt";
    std::string validation_filename = "data/validation.txt";
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

    model.Train(X_train, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY, X_test, y_test, false);
}
#endif