#include "radio_train.h"

int radio_train_main()
{
	constexpr int NUM_CLASSES = 4;
	constexpr int SAMPLES_PER_FRAME = 4096;
	constexpr int IQ_PAIRS = 2048;
	constexpr int IQ_CHANNELS = 2;
	constexpr int DENSE_INPUT_SIZE = 64 * 64;


	const std::vector<std::string> CLASS_NAMES = {"BPSK", "QPSK", "16-QAM", "32-QAM"};
	std::cout << "Loading GNU Radio dataset..." << std::endl;

	Eigen::MatrixXd train_data, test_data;
	Eigen::VectorXi train_labels, test_labels;

	try
	{
		NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/pluto_train_data.csv", train_data);
		NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/pluto_train_labels.csv", train_labels);
	
		NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/pluto_test_data.csv", test_data);
		NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/pluto_test_labels.csv", test_labels);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to load dataset: " << e.what() << std::endl;
		return 1;
	}

	Eigen::MatrixXd y_train(train_labels.size(), 1);
	Eigen::MatrixXd y_test(test_labels.size(), 1);
	for (int i = 0; i < train_labels.size(); i++)
	{
		y_train(i, 0) = static_cast<double>(train_labels(i));
	}
	for (int i = 0; i < test_labels.size(); i++)
	{
		y_test(i, 0) = static_cast<double>(test_labels(i));
	}

	if (train_data.cols() != SAMPLES_PER_FRAME)
	{
		std::cerr << "ERROR: Expected " << SAMPLES_PER_FRAME
				  << " features per sample, got " << train_data.cols() << std::endl;
		return -1;
	}

	std::cout << "Dataset loaded successfully!" << std::endl;
	std::cout << "  Training samples: " << train_data.rows() << std::endl;
	std::cout << "  Test samples: " << test_data.rows() << std::endl;

	try
	{
		NEURAL_NETWORK::Helpers::ShuffleData(train_data, y_train);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to shuffle data: " << e.what() << std::endl;
		return 1;
	}

	NEURAL_NETWORK::Model model;
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());

	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		16,
		7,
		IQ_PAIRS,
		IQ_CHANNELS,
		1,
		1,
		0.0, 5e-4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(16));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		4,
		IQ_PAIRS,
		16,
		4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));

	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		32,
		5,
		512,
		16,
		1,
		1,
		0.0, 5e-4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(32));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		4,
		512,
		32,
		4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.3));

	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		64,
		3,
		128,
		32,
		1,
		1,
		0.0, 5e-4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(64));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		2,
		128,
		64,
		2
	));
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.4));

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		DENSE_INPUT_SIZE,
		128,
		0.0, 5e-4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5));

	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		128,
		NUM_CLASSES,
		0.0, 5e-4
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.0005, 1e-5)
	);

	model.Finalize();

	std::cout << "Starting training..." << std::endl;
	std::cout << std::endl;

	model.Train(train_data, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY,
				SAVE_EVERY, test_data, y_test);

	std::cout << std::endl;
	std::cout << "Saving trained model..." << std::endl;
	model.SaveModel("../data/rf_modulation_classifier.bin");

	std::cout << "Final evaluation on test set:" << std::endl;
	model.Evaluate(test_data, y_test, BATCH_SIZE);

	std::cout << "Example predictions (first 10 test samples):" << std::endl;
	constexpr int NUM_EXAMPLES = 10;
	Eigen::MatrixXd sample_data = test_data.topRows(NUM_EXAMPLES);
	Eigen::MatrixXd sample_labels = y_test.topRows(NUM_EXAMPLES);
	Eigen::MatrixXd predictions = model.Predict(sample_data, 1);

	int correct = 0;
	for (int i = 0; i < NUM_EXAMPLES; i++)
	{
		int predicted = static_cast<int>(predictions(i, 0));
		int actual = static_cast<int>(sample_labels(i, 0));

		if ((predicted == actual))
		{
			correct++;
		}
		
		std::cout << "  Sample " << i << ": "
				  << "Predicted=" << CLASS_NAMES[predicted]
				  << ", Actual=" << CLASS_NAMES[actual]
				  << std::endl;
	}
	std::cout << "  Accuracy on examples: " << correct << "/" << NUM_EXAMPLES << std::endl;

	return 0;
}