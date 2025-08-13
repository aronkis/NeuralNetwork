#include <iostream>
#include <Eigen/Dense>
// #include <matplot/matplot.h>
#include "LayerDense.h"
#include "LayerDropout.h"
#include "Helpers.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "ActivationSigmoid.h"
#include "LossCategoricalCrossEntropy.h"
#include "LossBinaryCrossEntropy.h"
#include "ActivationSoftmaxLossCategoricalCrossEntropy.h"
#include "StochasticGradientDescent.h"
#include "AdaGrad.h"
#include "RMSProp.h"
#include "Adam.h"

// #define CLASSIFICATION 1
#define BINARY_CLASSIFICATION 1

#ifdef CLASSIFICATION
int main() {

	std::string filename = "../data/points.txt"; 

	Eigen::MatrixXd X;      
	Eigen::MatrixXi y;      

	NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X, y);

	if (X.rows() > 0) 
	{
		std::cout << "Successfully read " << X.rows() << " points." << std::endl;
	} 
	else 
	{
		std::cout << "Could not read any data or file not found." << std::endl;
	}

	NEURAL_NETWORK::LayerDense l1(2, 64, 0.0, 5e-4, 0.0, 5e-4);
	
	NEURAL_NETWORK::ActivationReLU activation_relu;

	NEURAL_NETWORK::LayerDropout l_d(0.1);

	NEURAL_NETWORK::LayerDense l2(64, 3);

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy loss_softmax;

	// NEURAL_NETWORK::StochasticGradientDescent optimizer(1.0, 1e-4, 0.81);
	// NEURAL_NETWORK::AdaGrad optimizer(0.7, 0.5e-4, 1e-9);
	// NEURAL_NETWORK::RMSProp optimizer(0.001, 0.5e-4, 1e-7, 0.7);
	// NEURAL_NETWORK::Adam optimizer(0.05, 5e-7, 0.9, 0.999, 1e-7);
	NEURAL_NETWORK::Adam optimizer(0.05, 5e-5);

	double accuracy = 0.0;

	for (int epoch = 0; epoch < 10001; epoch++) 
	{
		l1.forward(X);
		activation_relu.forward(l1.GetOutput());
		l_d.forward(activation_relu.GetOutput());
		l2.forward(l_d.GetOutput());
		loss_softmax.forward(l2.GetOutput(), y);

		double data_loss = loss_softmax.GetLoss();
		double reg_loss = loss_softmax.GetLossFunction().RegularizationLoss(l1) + 
						 loss_softmax.GetLossFunction().RegularizationLoss(l2);
		double loss = data_loss + reg_loss;
		
		if (epoch % 1000 == 0) 
		{
			accuracy = NEURAL_NETWORK::Helpers::CalculateAccuracy(loss_softmax.GetOutput(), y);
			std::cout << "Epoch: " << epoch 
					  << ", Accuracy: " << accuracy
					  << ", Loss: " << loss 
					  << ", Data loss: " << data_loss
					  << ", Regularization loss: " << reg_loss
					  << ", Learning rate: " << optimizer.GetLearningRate()
					  << std::endl;
		}

		loss_softmax.backward(loss_softmax.GetOutput(), y);
		l2.backward(loss_softmax.GetDInput());
		l_d.backward(l2.GetDInput());
		activation_relu.backward(l_d.GetDInput());
		l1.backward(activation_relu.GetDInput());

		optimizer.PreUpdateParameters();
		optimizer.UpdateParameters(l1);
		optimizer.UpdateParameters(l2);
		optimizer.PostUpdateParameters();
	}

	std::string validation_filename = "../data/validation.txt"; 

	Eigen::MatrixXd X_test;      
	Eigen::MatrixXi y_test;      

	NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

	l1.forward(X_test);
	activation_relu.forward(l1.GetOutput());
	l2.forward(activation_relu.GetOutput());

	loss_softmax.forward(l2.GetOutput(), y_test);

	accuracy = NEURAL_NETWORK::Helpers::CalculateAccuracy(loss_softmax.GetOutput(), y_test);
	std::cout << "Validation Accuracy: " << accuracy << ", Loss: " << loss_softmax.GetLoss() << std::endl;

	return 0;
}

#endif

#ifdef BINARY_CLASSIFICATION
int main() {

	std::string filename = "../data/binary.txt"; 

	Eigen::MatrixXd X;      
	Eigen::MatrixXi y;      

	NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X, y);

	if (X.rows() > 0) 
	{
		std::cout << "Successfully read " << X.rows() << " points." << std::endl;
	} 
	else 
	{
		std::cout << "Could not read any data or file not found." << std::endl;
	}

	NEURAL_NETWORK::LayerDense l1(2, 64, 0.0, 5e-4, 0.0, 5e-4);
	
	NEURAL_NETWORK::ActivationReLU activation_relu;

	NEURAL_NETWORK::LayerDense l2(64, 1);

	NEURAL_NETWORK::ActivationSigmoid activation_sigmoid;

	NEURAL_NETWORK::LossBinaryCrossEntropy loss_binary_cross_entropy;

	NEURAL_NETWORK::Adam optimizer(0.019, 5e-7);

	double accuracy = 0.0;

	for (int epoch = 0; epoch < 10001; epoch++) 
	{
		l1.forward(X);
		activation_relu.forward(l1.GetOutput());
		l2.forward(activation_relu.GetOutput());
		activation_sigmoid.forward(l2.GetOutput());
		loss_binary_cross_entropy.CalculateLoss(activation_sigmoid.GetOutput(), y);

		double data_loss = loss_binary_cross_entropy.GetLoss();
		double reg_loss = loss_binary_cross_entropy.RegularizationLoss(l1) + 
						  loss_binary_cross_entropy.RegularizationLoss(l2);
		double loss = data_loss + reg_loss;

		Eigen::MatrixXi prediction = (activation_sigmoid.GetOutput().array() > 0.5).cast<int>();
		accuracy = (prediction.array() == y.array()).cast<double>().mean();
		if (epoch % 500 == 0) 
		{
			std::cout << "Epoch: " << epoch 
					  << ", Accuracy: " << accuracy
					  << ", Loss: " << loss 
					  << " (Data loss: " << data_loss
					  << " | Regularization loss: " << reg_loss
					  << "), Learning rate: " << optimizer.GetLearningRate()
					  << std::endl;
		}

		loss_binary_cross_entropy.backward(activation_sigmoid.GetOutput(), y);
		activation_sigmoid.backward(loss_binary_cross_entropy.GetDInput());
		l2.backward(activation_sigmoid.GetDInput());
		activation_relu.backward(l2.GetDInput());
		l1.backward(activation_relu.GetDInput());

		optimizer.PreUpdateParameters();
		optimizer.UpdateParameters(l1);
		optimizer.UpdateParameters(l2);
		optimizer.PostUpdateParameters();
	}

	std::string validation_filename = "../data/binary_validation.txt"; 

	Eigen::MatrixXd X_test;      
	Eigen::MatrixXi y_test;      

	NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

	l1.forward(X_test);
	activation_relu.forward(l1.GetOutput());
	l2.forward(activation_relu.GetOutput());
	activation_sigmoid.forward(l2.GetOutput());
	loss_binary_cross_entropy.CalculateLoss(activation_sigmoid.GetOutput(), y_test);


	Eigen::MatrixXi prediction = (activation_sigmoid.GetOutput().array() > 0.5).cast<int>();
	accuracy = (prediction.array() == y.array()).cast<double>().mean();

	std::cout << "Validation Accuracy: " << accuracy << ", Loss: " << loss_binary_cross_entropy.GetLoss() << std::endl;

	return 0;
}
#endif