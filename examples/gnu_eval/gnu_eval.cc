#include "gnu_eval.h"
#include "ZMQ.h"
#include <iomanip>
#include <cmath>

int gnu_eval_main() 
{
	constexpr int IQ_PAIRS = 2048;
	constexpr int SAMPLES_PER_FRAME = 4096;

	const std::vector<std::string> CLASS_NAMES = {"BPSK", "QPSK", "16-QAM", "32-QAM"};


	constexpr int DISPLAY_UPDATE_MS = 500;
	std::cout << "Loading trained model..." << std::endl;
	NEURAL_NETWORK::Model model;
	
	try 
	{
		model.LoadModel("../data/rf_modulation_classifier.bin");
		std::cout << "Model loaded successfully!" << std::endl;
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "Failed to load model: " << e.what() << std::endl;
		std::cerr << "  Make sure to train the model first with radio_train" << std::endl;
		return 1;
	}


	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();

	zmq.AddOptions(ZMQ_CONFLATE, 1);
	zmq.AddOptions(ZMQ_RCVHWM, 2);

	const char* zmq_address = "tcp://127.0.0.1:5555";
	zmq.Connect(zmq_address);
	zmq.SubscribeToAllMessages();

	std::cout << "Connected to " << zmq_address << std::endl;
	std::cout << "Waiting for data from GNU Radio..." << std::endl;
	std::cout << "Press Ctrl+C to stop." << std::endl;

	auto start_time = std::chrono::steady_clock::now();
	auto last_display_time = start_time;

	while (zmq.Running()) 
	{
		if (!zmq.ReceiveMessage(ZMQ_DONTWAIT))
		{
			std::this_thread::sleep_for(std::chrono::microseconds(100));
			continue;
		}

		auto now = std::chrono::steady_clock::now();
		auto time_since_display = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_display_time).count();
		
		if (time_since_display < DISPLAY_UPDATE_MS)
		{
			continue;
		}

		last_display_time = now;

		size_t num_complex_samples = zmq.GetMessageSize() / sizeof(std::complex<float>);
		
		if (num_complex_samples < IQ_PAIRS)
		{
			continue;
		}

		std::complex<float>* samples = static_cast<std::complex<float>*>(zmq.GetMessageData());

		Eigen::MatrixXd input_frame(1, SAMPLES_PER_FRAME);

		for (size_t i = 0; i < IQ_PAIRS; i++) 
		{
			input_frame(0, i * 2)     = static_cast<double>(samples[i].real());
			input_frame(0, i * 2 + 1) = static_cast<double>(samples[i].imag());
		}

		double max_abs = input_frame.cwiseAbs().maxCoeff();
		if (max_abs > 1e-10)
		{
			input_frame /= max_abs;
		}

		Eigen::MatrixXd prediction = model.Predict(input_frame, 1);
		int predicted_class = static_cast<int>(prediction(0, 0));

		std::cout << "Predicted: " << std::setw(8) << CLASS_NAMES[predicted_class] << std::endl;
		std::cout << std::flush;
	}

	return 0;
}
