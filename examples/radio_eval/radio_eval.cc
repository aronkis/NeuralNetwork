#include "radio_eval.h"

int radio_eval_main()
{
	std::cout << "\nLoading test dataset..." << std::endl;
	Eigen::MatrixXd test_data;
	Eigen::VectorXi test_labels;

	try
	{
		NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/pluto_test_data.csv", test_data);
		NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/pluto_test_labels.csv", test_labels);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load test data: " << e.what() << std::endl;
		return 1;
	}

	Eigen::MatrixXd y_test(test_labels.size(), 1);
	for (int i = 0; i < test_labels.size(); i++)
	{
		y_test(i, 0) = static_cast<double>(test_labels(i));
	}

	std::cout << "Test samples: " << test_data.rows() << std::endl;

	std::cout << "\nLoading trained model..." << std::endl;
	NEURAL_NETWORK::Model model;
	try
	{
		model.LoadModel("../models/rf_modulation_classifier.bin");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load model: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Model loaded successfully!" << std::endl;

	const std::vector<std::string> class_names = {"BPSK", "QPSK", "16-QAM", "32-QAM"};

	std::cout << "\nEvaluating model on test data:" << std::endl;
	model.Evaluate(test_data, y_test, BATCH_SIZE);

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
				  << std::endl;
	}

	return 0;
}