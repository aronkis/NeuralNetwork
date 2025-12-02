#ifndef __ZMQ_H__
#define __ZMQ_H__

#include <zmq.h>

namespace NEURAL_NETWORK
{
	
	class ZMQ
	{
	public:
		ZMQ();
		~ZMQ();

		ZMQ(const ZMQ&) = delete;
		ZMQ& operator=(const ZMQ&) = delete;

		void CreateSubscriber();
		void Connect(const char* address);
		void SubscribeToAllMessages();
		bool ReceiveMessage(int flags = ZMQ_DONTWAIT);
		bool Running() const;
		size_t GetMessageSize();
		void* GetMessageData();

		static void signal_handler(int signum);

	private:
		void* context_;
		void* subscriber_;
		zmq_msg_t message_;
		void* data_;
		size_t size_;
		static volatile sig_atomic_t stop_;

	};

}

#endif // __ZMQ_H__