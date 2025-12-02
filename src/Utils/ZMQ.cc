#include "ZMQ.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

volatile sig_atomic_t NEURAL_NETWORK::ZMQ::stop_ = 0;

void NEURAL_NETWORK::ZMQ::signal_handler(int signum) 
{
    stop_ = 1;
}

NEURAL_NETWORK::ZMQ::ZMQ()
{
	std::cout << "\nConnecting to ZeroMQ..." << std::endl;
	context_ = zmq_ctx_new();
    if (!context_) 
    {
		std::cerr << "Failed to create ZMQ context" << std::endl;
        return;
    }
	stop_ = 0;
	subscriber_ = nullptr;
	data_ = nullptr;
	size_ = 0;
	signal(SIGINT, signal_handler);
}

NEURAL_NETWORK::ZMQ::~ZMQ()
{
	if (subscriber_) 
	{
		zmq_close(subscriber_);
	}

	if (context_) 
	{
		zmq_ctx_destroy(context_);
	}

	if (data_)
    {
        delete[] static_cast<char*>(data_);
    }
}

void NEURAL_NETWORK::ZMQ::CreateSubscriber()
{
	subscriber_ = zmq_socket(context_, ZMQ_SUB);
	if (!subscriber_) 
    {
        std::cerr << "Failed to create socket" << std::endl;
        zmq_ctx_destroy(context_);
        return;
    }
}

void NEURAL_NETWORK::ZMQ::Connect(const char* address)
{
	int rc = zmq_connect(subscriber_, address);
	if (rc != 0) 
	{
		std::cerr << "Failed to connect: " << zmq_strerror(errno) << std::endl;
		zmq_close(subscriber_);
		zmq_ctx_destroy(context_);
		return;
	}
}

void NEURAL_NETWORK::ZMQ::SubscribeToAllMessages()
{
	int rc = zmq_setsockopt(subscriber_, ZMQ_SUBSCRIBE, "", 0);
	if (rc != 0) 
	{
		std::cerr << "Failed to subscribe: " << zmq_strerror(errno) << std::endl;
		zmq_close(subscriber_);
		zmq_ctx_destroy(context_);
		return;
	}
	int conflate = 1;
    zmq_setsockopt(subscriber_, ZMQ_CONFLATE, &conflate, sizeof(conflate));
	
	int hwm = 2;
    zmq_setsockopt(subscriber_, ZMQ_RCVHWM, &hwm, sizeof(hwm));
}

bool NEURAL_NETWORK::ZMQ::ReceiveMessage(int flags)
{
    zmq_msg_t message;
    int rc = zmq_msg_init(&message);
    if (rc != 0) 
    {
        std::cerr << "Failed to initialize message: " << zmq_strerror(errno) << std::endl;
        return false;
    }
    
    rc = zmq_recvmsg(subscriber_, &message, flags);
    if (rc == -1) 
    {
        zmq_msg_close(&message);
        if (errno != EAGAIN && errno != EINTR) 
        {
            std::cerr << "Error receiving: " << zmq_strerror(errno) << std::endl;
        }
        return false;
    }

    size_ = zmq_msg_size(&message);

	if (data_)
    {
        delete[] static_cast<char*>(data_);
        data_ = nullptr;
    }

	if (size_ > 0)
    {
        data_ = new char[size_];
        std::memcpy(data_, zmq_msg_data(&message), size_);
    }

    zmq_msg_close(&message);
	return true;

}

bool NEURAL_NETWORK::ZMQ::Running() const
{
	return stop_ == 0;
}

size_t NEURAL_NETWORK::ZMQ::GetMessageSize()
{
	return size_;
}

void* NEURAL_NETWORK::ZMQ::GetMessageData()
{
	return data_;
}