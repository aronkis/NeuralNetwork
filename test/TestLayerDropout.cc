#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "LayerDropout.h"

class LayerDropoutTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		inputs = Eigen::MatrixXd::Ones(4, 3);
	}

	Eigen::MatrixXd inputs;
};

TEST_F(LayerDropoutTest, ForwardTrainingScalesCorrectly)
{
	NEURAL_NETWORK::LayerDropout dropout(0.5);
	dropout.forward(inputs, true);

	const auto& output = dropout.GetOutput();

	for (int r = 0; r < output.rows(); r++)
	{
		for (int c = 0; c < output.cols(); c++)
		{
			double value = output(r, c);
			EXPECT_TRUE(value == 0.0 || std::abs(value - 2.0) < 1e-9)
				<< "Unexpected dropout scaling";
		}
	}
}

TEST_F(LayerDropoutTest, ForwardInferenceBypassesMask)
{
	NEURAL_NETWORK::LayerDropout dropout(0.25);
	dropout.forward(inputs, false);

	const auto& output = dropout.GetOutput();
	EXPECT_TRUE(output.isApprox(inputs, 1e-12));
}

TEST_F(LayerDropoutTest, BackwardUsesMask)
{
	NEURAL_NETWORK::LayerDropout dropout(0.5);
	dropout.forward(inputs, true);

	Eigen::MatrixXd dvalues = Eigen::MatrixXd::Ones(inputs.rows(), inputs.cols());
	dropout.backward(dvalues);

	const auto& gradients = dropout.GetDInput();

	for (int r = 0; r < gradients.rows(); r++)
	{
		for (int c = 0; c < gradients.cols(); c++)
		{
			double g = gradients(r, c);
			EXPECT_TRUE(g == 0.0 || std::abs(g - 2.0) < 1e-9)
				<< "Unexpected dropout backward scaling";
		}
	}
}