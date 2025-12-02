#include "ZMQWrapper.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

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
		throw std::runtime_error("Failed to create ZMQ context");
	}
	stop_ = 0;
	subscriber_ = nullptr;
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
}

void NEURAL_NETWORK::ZMQ::CreateSubscriber()
{
	subscriber_ = zmq_socket(context_, ZMQ_SUB);
	if (!subscriber_) 
	{
		throw std::runtime_error("Failed to create ZMQ socket");
	}
}

void NEURAL_NETWORK::ZMQ::Connect(const char* address)
{
	int rc = zmq_connect(subscriber_, address);
	if (rc != 0) 
	{
		throw std::runtime_error(std::string("Failed to connect to ZMQ: ") + zmq_strerror(errno));
	}
}

void NEURAL_NETWORK::ZMQ::SubscribeToAllMessages()
{
	int rc = zmq_setsockopt(subscriber_, ZMQ_SUBSCRIBE, "", 0);
	if (rc != 0) 
	{
		throw std::runtime_error(std::string("Failed to subscribe to ZMQ: ") + zmq_strerror(errno));
	}
}

void NEURAL_NETWORK::ZMQ::AddOptions(int option, int value)
{
	int rc = zmq_setsockopt(subscriber_, option, &value, sizeof(value));
	if (rc != 0) 
	{
		throw std::runtime_error(std::string("Failed to set ZMQ option: ") + zmq_strerror(errno));
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

	if (msg_size > 0)
	{
		data_.resize(msg_size);
		std::memcpy(data_.data(), msg_data, msg_size);
	}
	else
	{
		data_.clear();
	}

	zmq_msg_close(&message_);
	return true;
}

bool NEURAL_NETWORK::ZMQ::Running() const noexcept
{
	return !stop_;
}

bool NEURAL_NETWORK::ZMQ::IsValid() const noexcept
{
	return context_ && subscriber_;
}

size_t NEURAL_NETWORK::ZMQ::GetMessageSize() const noexcept
{
	return data_.size();
}

void* NEURAL_NETWORK::ZMQ::GetMessageData() noexcept
{
	return data_.empty() ? nullptr : data_.data();
}