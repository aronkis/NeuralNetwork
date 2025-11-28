#include "gnu_eval.h"
#include <chrono>
#include <thread>
#include <iomanip>
#include <cmath>

constexpr int IQ_PAIRS = 2048;           // Complex samples per frame
constexpr int SAMPLES_PER_FRAME = 4096;  // Total floats (I/Q interleaved)

// Class names for display
const std::vector<std::string> CLASS_NAMES = {"BPSK", "QPSK", "16-QAM", "32-QAM"};

// Display update interval (milliseconds) - only affects display, not consumption
constexpr int DISPLAY_UPDATE_MS = 500;

// Enable debug output to diagnose signal statistics
constexpr bool DEBUG_SIGNAL_STATS = false;

// Global flag for interrupt handling
volatile sig_atomic_t stop = 0;

void signal_handler(int signum) 
{
    stop = 1;
}

int gnu_eval_main() 
{
    std::cout << "Loading trained model..." << std::endl;
    NEURAL_NETWORK::Model model;
    
    try 
    {
        model.LoadModel("../data/rf_modulation_classifier_old.bin");
        std::cout << "Model loaded successfully!" << std::endl;
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "  Make sure to train the model first with radio_train" << std::endl;
        return 1;
    }

    std::cout << "\nConnecting to ZeroMQ..." << std::endl;
    
    // Register signal handler
    signal(SIGINT, signal_handler);

    // Create ZMQ context
    void *context = zmq_ctx_new();
    if (!context) 
    {
        std::cerr << "Failed to create ZMQ context" << std::endl;
        return 1;
    }

    // Create SUB socket
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    if (!subscriber) 
    {
        std::cerr << "Failed to create socket" << std::endl;
        zmq_ctx_destroy(context);
        return 1;
    }

    // Connect to GNU Radio ZeroMQ publisher
    const char* zmq_address = "tcp://127.0.0.1:5555";
    int rc = zmq_connect(subscriber, zmq_address);
    if (rc != 0) 
    {
        std::cerr << "Failed to connect: " << zmq_strerror(errno) << std::endl;
        zmq_close(subscriber);
        zmq_ctx_destroy(context);
        return 1;
    }

    // Subscribe to all messages
    rc = zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
    if (rc != 0) 
    {
        std::cerr << "Failed to subscribe: " << zmq_strerror(errno) << std::endl;
        zmq_close(subscriber);
        zmq_ctx_destroy(context);
        return 1;
    }

    int hwm = 2;
    zmq_setsockopt(subscriber, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    
    std::cout << "Connected to " << zmq_address << std::endl;
    std::cout << "Expecting " << IQ_PAIRS << " complex samples per frame (" << SAMPLES_PER_FRAME << " floats)" << std::endl;
    std::cout << "Waiting for data from GNU Radio..." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    auto last_display_time = start_time - std::chrono::milliseconds(DISPLAY_UPDATE_MS); // Force immediate first display
    bool first_frame = true;

    while (!stop) 
    {
        zmq_msg_t message;
        zmq_msg_init(&message);

        int len = zmq_recvmsg(subscriber, &message, ZMQ_DONTWAIT);
        
        if (len == -1) 
        {
            if (errno == EAGAIN) 
            {
                zmq_msg_close(&message);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            if (errno == EINTR) 
            {
                zmq_msg_close(&message);
                break;
            }
            std::cerr << "Error receiving: " << zmq_strerror(errno) << std::endl;
            zmq_msg_close(&message);
            break;
        }

        auto now = std::chrono::steady_clock::now();
        auto time_since_display = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_display_time).count();
        
        if (time_since_display < DISPLAY_UPDATE_MS)
        {
            zmq_msg_close(&message);
            continue;
        }

        last_display_time = now;

        void* data = zmq_msg_data(&message);
        size_t size = zmq_msg_size(&message);
        
        size_t num_complex_samples = size / sizeof(std::complex<float>);
        
        // Print diagnostic info on first frame
        if (first_frame)
        {
            std::cout << "First frame received: " << size << " bytes = " 
                      << num_complex_samples << " complex samples" << std::endl;
            first_frame = false;
        }
        
        if (num_complex_samples < IQ_PAIRS) 
        {
            // Not enough samples, skip this frame
            if (DEBUG_SIGNAL_STATS)
            {
                std::cout << "[DEBUG] Skipping frame: only " << num_complex_samples 
                          << " samples, need " << IQ_PAIRS << std::endl;
            }
            zmq_msg_close(&message);
            continue;
        }
        
        std::complex<float>* samples = static_cast<std::complex<float>*>(data);

        Eigen::MatrixXd input_frame(1, SAMPLES_PER_FRAME);
        
        // Step 1: Copy I/Q data to the input frame
        for (size_t i = 0; i < IQ_PAIRS; i++) 
        {
            input_frame(0, i * 2)     = static_cast<double>(samples[i].real());
            input_frame(0, i * 2 + 1) = static_cast<double>(samples[i].imag());
        }

        // Step 2: Normalize to [-1, +1] range (matching training data preprocessing)
        // The training data was normalized using: data / max(abs(data))
        double max_abs = input_frame.cwiseAbs().maxCoeff();
        if (max_abs > 1e-10)  // Avoid division by zero
        {
            input_frame /= max_abs;
        }

        // Debug: Print signal statistics to diagnose issues
        if (DEBUG_SIGNAL_STATS)
        {
            double mean = input_frame.mean();
            double min_val = input_frame.minCoeff();
            double max_val = input_frame.maxCoeff();
            double rms = std::sqrt(input_frame.array().square().mean());
            std::cout << "[DEBUG] Signal stats: mean=" << std::fixed << std::setprecision(4) << mean
                      << ", min=" << min_val << ", max=" << max_val
                      << ", rms=" << rms << ", raw_max=" << max_abs << std::endl;
        }

        Eigen::MatrixXd prediction = model.Predict(input_frame, 1);
        int predicted_class = static_cast<int>(prediction(0, 0));
        
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        std::cout << "\r";
        std::cout << "Predicted: " << std::setw(8) << CLASS_NAMES[predicted_class] << std::endl;
        std::cout << std::flush;

        zmq_msg_close(&message);
    }

    std::cout << "\nCleaning up..." << std::endl;
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    std::cout << "Done." << std::endl;

    return 0;
}
