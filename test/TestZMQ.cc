#include <gtest/gtest.h>
#include "ZMQWrapper.h"
#include <thread>
#include <chrono>

class ZMQTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
	}

	void TearDown() override 
	{
	}
};

TEST_F(ZMQTest, ContextCreation) 
{
	EXPECT_NO_THROW({
		NEURAL_NETWORK::ZMQ zmq;
	});
}

TEST_F(ZMQTest, CreateSubscriber) 
{
	NEURAL_NETWORK::ZMQ zmq;
	EXPECT_NO_THROW(zmq.CreateSubscriber());
	EXPECT_TRUE(zmq.IsValid());
}

TEST_F(ZMQTest, AddOptions) 
{
	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();
	
	EXPECT_NO_THROW(zmq.AddOptions(ZMQ_CONFLATE, 1));
	EXPECT_NO_THROW(zmq.AddOptions(ZMQ_RCVHWM, 2));
}

TEST_F(ZMQTest, ConnectValidAddress) 
{
	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();
	
	EXPECT_NO_THROW(zmq.Connect("tcp://127.0.0.1:55555"));
}

TEST_F(ZMQTest, ConnectInvalidAddress) 
{
	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();
	
	EXPECT_THROW(zmq.Connect("invalid://address"), std::runtime_error);
}

TEST_F(ZMQTest, SubscribeToAllMessages) 
{
	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();
	
	EXPECT_NO_THROW(zmq.SubscribeToAllMessages());
}

TEST_F(ZMQTest, FullSetupSequence) 
{
	NEURAL_NETWORK::ZMQ zmq;
	
	EXPECT_NO_THROW({
		zmq.CreateSubscriber();
		zmq.AddOptions(ZMQ_CONFLATE, 1);
		zmq.AddOptions(ZMQ_RCVHWM, 2);
		zmq.Connect("tcp://127.0.0.1:55556");
		zmq.SubscribeToAllMessages();
	});
	
	EXPECT_TRUE(zmq.IsValid());
}

TEST_F(ZMQTest, RunningFlag) 
{
	NEURAL_NETWORK::ZMQ zmq;
	
	EXPECT_TRUE(zmq.Running());
	
	NEURAL_NETWORK::ZMQ::Stop();
	EXPECT_FALSE(zmq.Running());
}

TEST_F(ZMQTest, ReceiveMessageNoData) 
{
	NEURAL_NETWORK::ZMQ zmq;
	zmq.CreateSubscriber();
	zmq.Connect("tcp://127.0.0.1:55557");
	zmq.SubscribeToAllMessages();
	
	bool received = zmq.ReceiveMessage(ZMQ_DONTWAIT);
	EXPECT_FALSE(received);
}

TEST_F(ZMQTest, MessageDataInitialState) 
{
	NEURAL_NETWORK::ZMQ zmq;
	
	EXPECT_EQ(zmq.GetMessageSize(), 0u);
	EXPECT_EQ(zmq.GetMessageData(), nullptr);
}

TEST_F(ZMQTest, SetSignalHandler) 
{
	NEURAL_NETWORK::ZMQ zmq;
	
	auto customHandler = [](int) {};
	EXPECT_NO_THROW(zmq.SetSignalHandler(customHandler, SIGINT));
}

TEST_F(ZMQTest, MultipleInstances) 
{
	EXPECT_NO_THROW({
		NEURAL_NETWORK::ZMQ zmq1;
		NEURAL_NETWORK::ZMQ zmq2;
		
		zmq1.CreateSubscriber();
		zmq2.CreateSubscriber();
		
		EXPECT_TRUE(zmq1.IsValid());
		EXPECT_TRUE(zmq2.IsValid());
	});
}
