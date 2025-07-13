#include <iostream>
#include <Eigen/Dense>
// #include <matplot/matplot.h>

#include "LayerDense.h"
#include "Helpers.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"

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

	NEURAL_NETWORK::LayerDense l1(2, 3);
	l1.forward(X);
	
	NEURAL_NETWORK::ActivationReLU activation_relu;
	activation_relu.forward(l1.GetOutput());
	
	NEURAL_NETWORK::LayerDense l2(3, 3);
	l2.forward(activation_relu.GetOutput());

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy loss_softmax;
	loss_softmax.forward(l2.GetOutput(), y);
	std::cout << "Loss: " << loss_softmax.GetOutput().topRows(5) << std::endl;
	std::cout << "Loss: " << loss_softmax.GetLoss() << std::endl;

	double accuracy = NEURAL_NETWORK::Helpers::CalculateAccuracy(loss_softmax.GetOutput(), y);
	std::cout << "Accuracy: " << accuracy << std::endl;

	loss_softmax.backward(loss_softmax.GetOutput(), y);
	l2.backward(loss_softmax.GetDInputs());
	activation_relu.backward(l2.GetDInput());
	l1.backward(activation_relu.GetDInput());

	std::cout << "\nWeights of L1 after backward pass:\n" 
			  << l1.GetDWeights() << std::endl;

	std::cout << "\nBiases of L1 after backward pass:\n"
				<< l1.GetDBiases() << std::endl;

	std::cout << "\nWeights of L2 after backward pass:\n" 
			  << l2.GetDWeights() << std::endl;

	std::cout << "\nBiases of L2 after backward pass:\n"
				<< l2.GetDBiases() << std::endl;

	return 0;
}