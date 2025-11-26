#include "cnn_eval.h"

int cnn_eval_main()
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

	return 0;
}
