#include "radio_eval.h"

int radio_eval_main()
{
	std::cout << "=== Radio Modulation Classification Evaluation ===" << std::endl;

	// Load test dataset
	std::cout << "\nLoading test dataset..." << std::endl;
	Eigen::MatrixXd test_data;
	Eigen::VectorXi test_labels;

	// Load the test CSV files
	NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/rf_modulation_test_data.csv", test_data);
	NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/rf_modulation_test_labels.csv", test_labels);

	// Convert labels to double matrix for framework compatibility
	Eigen::MatrixXd y_test(test_labels.size(), 1);
	for (int i = 0; i < test_labels.size(); i++)
	{
		y_test(i, 0) = static_cast<double>(test_labels(i));
	}

	std::cout << "Test samples: " << test_data.rows() << std::endl;
	std::cout << "Features per sample: " << test_data.cols() << std::endl;

	// Load the trained model
	std::cout << "\nLoading trained model..." << std::endl;
	NEURAL_NETWORK::Model model;
	model.LoadModel("../data/rf_modulation_classifier.bin");
	std::cout << "Model loaded successfully!" << std::endl;

	// Define modulation class names
	std::vector<std::string> class_names = {
		"BPSK", "QPSK", "8PSK", "16PSK", "64PSK",          // PSK Family (0-4)
		"16QAM", "32QAM", "64QAM",                         // QAM Family (5-7)
		"ASK", "2FSK", "4FSK", "8FSK",                     // Digital (8-11)
		"GMSK", "MSK", "AM", "FM"                          // Advanced (12-15)
	};

	// Evaluate the model on the entire test set
	std::cout << "\nEvaluating model on test data:" << std::endl;
	model.Evaluate(test_data, y_test, BATCH_SIZE);

	// Show predictions for first 10 samples
	std::cout << "\nFirst 10 test samples predictions:" << std::endl;
	Eigen::MatrixXd sample_data = test_data.topRows(10);
	Eigen::MatrixXd sample_labels = y_test.topRows(10);
	Eigen::MatrixXd predictions = model.Predict(sample_data, 1);

	for (int i = 0; i < 10; i++)
	{
		int predicted_class = static_cast<int>(predictions(i, 0));
		int actual_class = static_cast<int>(sample_labels(i, 0));
		std::cout << "Sample " << i << ": Predicted = " << class_names[predicted_class]
				  << ", Actual = " << class_names[actual_class]
				  << (predicted_class == actual_class ? " ✓" : " ✗") << std::endl;
	}

	return 0;
}