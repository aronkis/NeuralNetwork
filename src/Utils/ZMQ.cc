#include "ZMQ.h"

#include <iostream>
#include <cstring>

volatile sig_atomic_t NEURAL_NETWORK::ZMQ::stop_ = 0;

void NEURAL_NETWORK::ZMQ::SignalHandler(int signum) 
{
	stop_ = 1;
}

void NEURAL_NETWORK::ZMQ::Stop()
{
	stop_ = 1;
}

void NEURAL_NETWORK::ZMQ::SetSignalHandler(SignalHandlerFunc handler, int signum)
{
	signal(signum, handler);
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
	signal(SIGINT, SignalHandler);
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
		return;
	}
}

void NEURAL_NETWORK::ZMQ::Connect(const char* address)
{
	int rc = zmq_connect(subscriber_, address);
	if (rc != 0) 
	{
		std::cerr << "Failed to connect: " << zmq_strerror(errno) << std::endl;
		return;
	}
}

void NEURAL_NETWORK::ZMQ::SubscribeToAllMessages()
{
	int rc = zmq_setsockopt(subscriber_, ZMQ_SUBSCRIBE, "", 0);
	if (rc != 0) 
	{
		std::cerr << "Failed to subscribe: " << zmq_strerror(errno) << std::endl;
		return;
	}
}

void NEURAL_NETWORK::ZMQ::AddOptions(int option, int value)
{
	int rc = zmq_setsockopt(subscriber_, option, &value, sizeof(value));
	if (rc != 0) 
	{
		std::cerr << "Failed to set option: " << zmq_strerror(errno) << std::endl;
		return;
	}
}

bool NEURAL_NETWORK::ZMQ::ReceiveMessage(int flags)
{
	int rc = zmq_msg_init(&message_);
	if (rc != 0) 
	{
		std::cerr << "Failed to initialize message: " << zmq_strerror(errno) << std::endl;
		return false;
	}
	
	rc = zmq_recvmsg(subscriber_, &message_, flags);
	if (rc == -1) 
	{
		zmq_msg_close(&message_);
		if (errno != EAGAIN && errno != EINTR) 
		{
			std::cerr << "Error receiving: " << zmq_strerror(errno) << std::endl;
		}
		return false;
	}

	size_t msg_size = zmq_msg_size(&message_);
	void* msg_data = zmq_msg_data(&message_);

	if (data_)
	{
		delete[] static_cast<char*>(data_);
		data_ = nullptr;
	}

	if (msg_size > 0)
	{
		data_ = new char[msg_size];
		std::memcpy(data_, msg_data, msg_size);
	}
	size_ = msg_size;

	zmq_msg_close(&message_);
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