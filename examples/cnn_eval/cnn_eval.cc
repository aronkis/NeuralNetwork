#include "cnn_eval.h"

int cnn_eval_main()
{
	std::cout << "BATCH_SIZE: " << BATCH_SIZE << std::endl;
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

	std::cout << "Loading model..." << std::endl;
	try
	{
		model.LoadModel("../models/fashion_mnist_CNN_complete_model.bin");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load model: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Model loaded successfully!" << std::endl;

	std::cout << "Evaluating on test data..." << std::endl;
	model.Evaluate(X_test, y_test, BATCH_SIZE);

	int num_samples_to_predict = 10;
	Eigen::MatrixXd sample_X = X_test.topRows(num_samples_to_predict);
	Eigen::MatrixXd sample_y = y_test.topRows(num_samples_to_predict);

	Eigen::MatrixXd predictions = model.Predict(sample_X, 1);

	std::cout << "Predictions vs Actual for first " << num_samples_to_predict << " samples:" << std::endl;
	for (int i = 0; i < num_samples_to_predict; i++)
	{
		int predicted_class = static_cast<int>(predictions(i, 0));
		int actual_class = static_cast<int>(sample_y(i, 0));
		std::cout << "Sample " << i << ": Predicted = " << predicted_class
				  << ", Actual = " << actual_class
				  << std::endl;
	}

	return 0;
}
