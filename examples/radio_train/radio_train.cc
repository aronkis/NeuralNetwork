#include "radio_train.h"

// =============================================================================
// Automatic Modulation Classification (AMC) from Raw RF I/Q Data
// =============================================================================
// Task: Classify the modulation scheme from raw baseband I/Q samples
// Target modulations: BPSK, QPSK, 16-QAM, 32-QAM (4 classes)
// Input format: 4096 samples per frame (2048 I/Q pairs, interleaved)
// Data source: GNU Radio VariableModulation.py via ZeroMQ (buff_size=2048)
//
// This is a classic AMC problem where the CNN learns to identify modulation
// schemes by extracting statistical and structural features from raw I/Q:
//   - Constellation geometry (number of symbols, grid structure)
//   - Higher-order statistics (kurtosis, cumulants)
//   - Temporal correlations and cyclostationary features
// =============================================================================

// Number of modulation classes
constexpr int NUM_CLASSES = 4;

// Input dimensions (matching VariableModulation.py buff_size=2048 complex samples)
constexpr int SAMPLES_PER_FRAME = 4096;  // Total samples (I/Q interleaved)
constexpr int IQ_PAIRS = 2048;           // Number of complex samples
constexpr int IQ_CHANNELS = 2;           // I and Q components

// Class label mapping (modulation schemes)
// 0 = BPSK   - Binary PSK (2 constellation points)
// 1 = QPSK   - Quadrature PSK (4 constellation points)
// 2 = 16-QAM - 16-point QAM (4x4 grid)
// 3 = 32-QAM - 32-point QAM (cross constellation)
const std::vector<std::string> CLASS_NAMES = {"BPSK", "QPSK", "16-QAM", "32-QAM"};

int radio_train_main()
{
	std::cout << "=== Automatic Modulation Classification (AMC) with 1D CNN ===" << std::endl;
	std::cout << "Task: Identify modulation scheme from raw I/Q samples" << std::endl;
	std::cout << "Target modulations: BPSK, QPSK, 16-QAM, 32-QAM" << std::endl;
	std::cout << "Input: " << SAMPLES_PER_FRAME << " samples/frame ("
	          << IQ_PAIRS << " I/Q pairs)" << std::endl;
	std::cout << std::endl;
	std::cout << "Training parameters:" << std::endl;
	std::cout << "  Epochs: " << NN_EPOCHS << std::endl;
	std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
	std::cout << "  Print every: " << NN_PRINT_EVERY << " steps" << std::endl;
	std::cout << std::endl;

	// =========================================================================
	// Load Dataset
	// =========================================================================
	// Expected CSV format from GNU Radio:
	//   - train_data.csv: Each row = 2048 floats (I0,Q0,I1,Q1,...,I1023,Q1023)
	//   - train_labels.csv: Each row = modulation class label (0-3)
	// The data should include samples at various SNR levels for robustness
	// =========================================================================
	std::cout << "Loading GNU Radio synthesized dataset..." << std::endl;

	Eigen::MatrixXd train_data, test_data;
	Eigen::VectorXi train_labels, test_labels;

	// Load training data
	NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/rf_modulation_train_data.csv", train_data);
	NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/rf_modulation_train_labels.csv", train_labels);

	// Load test/validation data
	NEURAL_NETWORK::Helpers::ReadCSVMatrix("../data/RF/Mod4/rf_modulation_test_data.csv", test_data);
	NEURAL_NETWORK::Helpers::ReadCSVLabels("../data/RF/Mod4/rf_modulation_test_labels.csv", test_labels);

	// Convert labels to double matrix for framework compatibility
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

	// Validate input dimensions
	if (train_data.cols() != SAMPLES_PER_FRAME)
	{
		std::cerr << "ERROR: Expected " << SAMPLES_PER_FRAME
		          << " features per sample, got " << train_data.cols() << std::endl;
		return -1;
	}

	std::cout << "Dataset loaded successfully!" << std::endl;
	std::cout << "  Training samples: " << train_data.rows() << std::endl;
	std::cout << "  Test samples: " << test_data.rows() << std::endl;
	std::cout << "  Features per sample: " << train_data.cols() << " (I/Q interleaved)" << std::endl;
	std::cout << "  Classes: " << NUM_CLASSES << " (";
	for (int i = 0; i < NUM_CLASSES; i++)
	{
		std::cout << CLASS_NAMES[i];
		if (i < NUM_CLASSES - 1) std::cout << ", ";
	}
	std::cout << ")" << std::endl;
	std::cout << std::endl;

	// Shuffle training data
	NEURAL_NETWORK::Helpers::ShuffleData(train_data, y_train);

	// =========================================================================
	// Build 1D CNN Model for Automatic Modulation Classification
	// =========================================================================
	// The CNN learns to extract discriminative features from raw I/Q data:
	//   - Conv layers: Learn temporal patterns, phase/amplitude relationships
	//   - BatchNorm: Stabilizes training, allows higher learning rates
	//   - Pooling: Provides invariance to timing offsets
	//   - Dense layers: Combine features for final classification
	//
	// Key insight: Different modulations have distinct statistical signatures
	// in their I/Q representation that the CNN learns to recognize.
	// =========================================================================
	std::cout << "Building 1D CNN for Automatic Modulation Classification..." << std::endl;
	std::cout << "  Using simplified architecture to prevent overfitting" << std::endl;

	NEURAL_NETWORK::Model model;
	model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());

	// -------------------------------------------------------------------------
	// Simplified Convolutional Feature Extraction
	// -------------------------------------------------------------------------
	// Reduced model complexity to prevent overfitting on 4 classes
	// Key: Focus on robust features that generalize to real-world data

	// Conv Block 1: Extract low-level I/Q statistics
	// Input: 2048 time steps × 2 channels (I/Q)
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		16,          // Reduced: 16 filters (was 32)
		7,           // kernel size 7 - captures local I/Q relationships
		IQ_PAIRS,    // input_length = 2048
		IQ_CHANNELS, // input_channels = 2 (I/Q)
		1,           // stride
		1,           // padding (same)
		0.0, 5e-4    // Increased L2 regularization
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(16));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	// MaxPool 1: Downsample 2048 → 512
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		4,          // pool_size
		IQ_PAIRS,   // input_length = 2048
		16,         // input_channels
		4           // stride
	));

	// Dropout early to prevent co-adaptation
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.2));

	// Conv Block 2: Extract mid-level modulation features
	// Input: 512 time steps × 16 channels
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		32,        // Reduced: 32 filters (was 64)
		5,         // kernel size 5
		512,       // input_length (after pool: 2048/4 = 512)
		16,        // input_channels
		1,         // stride
		1,         // padding
		0.0, 5e-4  // Increased L2 regularization
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(32));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	// MaxPool 2: Downsample 512 → 128
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		4,          // pool_size
		512,        // input_length
		32,         // input_channels
		4           // stride
	));

	// Dropout after second conv block
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.3));

	// Conv Block 3: Extract high-level modulation signatures
	// Input: 128 time steps × 32 channels
	model.Add(std::make_shared<NEURAL_NETWORK::Convolution1D>(
		64,        // Reduced: 64 filters (was 128)
		3,         // kernel size 3
		128,       // input_length (after pool: 512/4 = 128)
		32,        // input_channels
		1,         // stride
		1,         // padding
		0.0, 5e-4  // Increased L2 regularization
	));
	model.Add(std::make_shared<NEURAL_NETWORK::BatchNormalization>(64));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());

	// MaxPool 3: Downsample 128 → 64
	model.Add(std::make_shared<NEURAL_NETWORK::MaxPooling1D>(
		BATCH_SIZE,
		2,          // pool_size
		128,        // input_length
		64,         // input_channels
		2           // stride
	));

	// Strong dropout for regularization
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.4));

	// -------------------------------------------------------------------------
	// Dense Classification Head (Simplified)
	// -------------------------------------------------------------------------
	// Flattened size: 64 time steps × 64 channels = 4,096 features
	// (2048 -> /4 -> 512 -> /4 -> 128 -> /2 -> 64)
	constexpr int DENSE_INPUT_SIZE = 64 * 64;

	// Dense 1: Compress features (smaller than before)
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		DENSE_INPUT_SIZE,
		128,       // Reduced: 128 neurons (was 256)
		0.0, 5e-4  // Increased L2 regularization
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationReLU>());
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDropout>(0.5));  // Increased dropout

	// Output layer: 4-class softmax for modulation classification
	// Removed intermediate dense layer to reduce overfitting
	model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(
		128,
		NUM_CLASSES,  // 4 modulation classes
		0.0, 5e-4     // Increased L2 regularization
	));
	model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

	std::cout << "Architecture summary (simplified to prevent overfitting):" << std::endl;
	std::cout << "  Input: " << SAMPLES_PER_FRAME << " (2048×2 I/Q)" << std::endl;
	std::cout << "  Conv1D(16, k=7) → BatchNorm → ReLU → MaxPool(4) → Dropout(0.2)" << std::endl;
	std::cout << "  Conv1D(32, k=5) → BatchNorm → ReLU → MaxPool(4) → Dropout(0.3)" << std::endl;
	std::cout << "  Conv1D(64, k=3) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.4)" << std::endl;
	std::cout << "  Dense(128) → ReLU → Dropout(0.5)" << std::endl;
	std::cout << "  Dense(" << NUM_CLASSES << ") → Softmax" << std::endl;
	std::cout << std::endl;

	// =========================================================================
	// Configure Optimizer and Train
	// =========================================================================
	// Lower learning rate for better generalization
	model.Set(
		std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
		std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
		std::make_unique<NEURAL_NETWORK::Adam>(0.0005, 1e-5)  // Reduced LR from 0.001 to 0.0005
	);

	model.Finalize();

	std::cout << "Starting training..." << std::endl;
	std::cout << "Random baseline accuracy: " << (100.0 / NUM_CLASSES) << "%" << std::endl;
	std::cout << std::endl;

	model.Train(train_data, y_train, BATCH_SIZE, NN_EPOCHS, NN_PRINT_EVERY,
	            SAVE_EVERY, test_data, y_test);

	// =========================================================================
	// Save and Evaluate
	// =========================================================================
	std::cout << std::endl;
	std::cout << "Saving trained model..." << std::endl;
	model.SaveModel("../data/rf_modulation_classifier.bin");
	std::cout << "Model saved to: data/rf_modulation_classifier.bin" << std::endl;

	std::cout << std::endl;
	std::cout << "Final evaluation on test set:" << std::endl;
	model.Evaluate(test_data, y_test, BATCH_SIZE);

	// Show example predictions
	std::cout << std::endl;
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
		bool is_correct = (predicted == actual);
		if (is_correct) correct++;

		std::cout << "  Sample " << i << ": "
		          << "Predicted=" << CLASS_NAMES[predicted]
		          << ", Actual=" << CLASS_NAMES[actual]
		          << (is_correct ? " ✓" : " ✗") << std::endl;
	}
	std::cout << "  Accuracy on examples: " << correct << "/" << NUM_EXAMPLES << std::endl;

	return 0;
}