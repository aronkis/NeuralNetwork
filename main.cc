#include <iostream>
#include <Eigen/Dense>
// #include <matplot/matplot.h>

#include "LayerDense.h"
#include "Helpers.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"
#include "StochasticGradientDescent.h"
#include "AdaGrad.h"
#include "RMSProp.h"
#include "Adam.h"

#ifdef METAL_ENABLED
#include "MetalDevice.h"
#include "MetalLayerDense.h"
#include "MetalActivationReLU.h"
#endif

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

#ifdef METAL_ENABLED
	// Check if Metal is available and initialize
	NEURAL_NETWORK::Metal::MetalDevice& metalDevice = NEURAL_NETWORK::Metal::MetalDevice::getInstance();
	bool useGPU = metalDevice.isMetalAvailable();
	
	// For small networks like this, CPU is faster than GPU
	// Only use GPU for larger networks (>1000 neurons or >10000 samples)
	bool networkTooSmall = (X.rows() < 10000) && (64 < 1000);  // 300 samples, 64 neurons
	if (networkTooSmall) {
		useGPU = false;
		std::cout << "Network too small for GPU acceleration, using CPU for better performance." << std::endl;
	}
	
	if (useGPU) {
		std::cout << "Metal GPU acceleration enabled!" << std::endl;
	} else if (metalDevice.isMetalAvailable()) {
		std::cout << "Metal available but using CPU for optimal performance." << std::endl;
	} else {
		std::cout << "Metal not available, using CPU." << std::endl;
	}
	
	// Use Metal-accelerated layers if beneficial
	NEURAL_NETWORK::Metal::MetalLayerDense l1(2, 64, useGPU);
	NEURAL_NETWORK::Metal::MetalActivationReLU activation_relu(useGPU);
	NEURAL_NETWORK::Metal::MetalLayerDense l2(64, 3, useGPU);
#else
	std::cout << "Metal support not compiled in, using CPU." << std::endl;
	
	NEURAL_NETWORK::LayerDense l1(2, 64);
	NEURAL_NETWORK::ActivationReLU activation_relu;
	NEURAL_NETWORK::LayerDense l2(64, 3);
#endif

	NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossentropy loss_softmax;

	// NEURAL_NETWORK::StochasticGradientDescent optimizer(1.0, 1e-4, 0.81);
	// NEURAL_NETWORK::AdaGrad optimizer(0.7, 0.5e-4, 1e-9);
	// NEURAL_NETWORK::RMSProp optimizer(0.001, 0.5e-4, 1e-7, 0.7);
	NEURAL_NETWORK::Adam optimizer(0.05, 5e-7, 0.9, 0.999, 1e-7);
	
	double accuracy = 0.0;

	for (int epoch = 0; epoch < 10001; epoch++) 
	{
		l1.forward(X);
		activation_relu.forward(l1.GetOutput());
		l2.forward(activation_relu.GetOutput());

		loss_softmax.forward(l2.GetOutput(), y);

		if (epoch % 1000 == 0) 
		{
			accuracy = NEURAL_NETWORK::Helpers::CalculateAccuracy(loss_softmax.GetOutput(), y);
			std::cout << "Epoch: " << epoch << ", Accuracy: " << accuracy<< ", Loss: " << loss_softmax.GetLoss() << std::endl;
		}

		loss_softmax.backward(loss_softmax.GetOutput(), y);
		l2.backward(loss_softmax.GetDInputs());
		activation_relu.backward(l2.GetDInput());
		l1.backward(activation_relu.GetDInput());

		optimizer.PreUpdateParameters();
		optimizer.UpdateParameters(l1);
		optimizer.UpdateParameters(l2);
		optimizer.PostUpdateParameters();
	}

	return 0;
}