#include <iostream>
#include <Eigen/Dense>
// #include <matplot/matplot.h>

#include "LayerDense.h"
#include "Helpers.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"

int main() {

    std::string filename = "../data/points.txt"; 

    Eigen::MatrixXd X;      
    Eigen::MatrixXi y;      

    Helpers::ReadSpiralIntoEigen(filename, X, y);

    if (X.rows() > 0) 
    {
        std::cout << "Successfully read " << X.rows() << " points." << std::endl;
    } 
    else 
    {
        std::cout << "Could not read any data or file not found." << std::endl;
    }

    LayerDense l1(2, 3);
    l1.forward(X);
    
    ActivationReLU activation_relu;
    activation_relu.forward(l1.GetOutput());
    
    LayerDense l2(3, 3);
    l2.forward(activation_relu.GetOutput());

    ActivationSoftmax activation_softmax;
    activation_softmax.forward(l2.GetOutput());
    
    std::cout << activation_softmax.GetOutput().topRows(5) << std::endl;

    LossCategoricalCrossentropy loss;
    loss.calculateLoss(activation_softmax.GetOutput(), y);
    std::cout << "Loss: " << loss.GetLoss() << std::endl;

    double accuracy = Helpers::CalculateAccuracy(activation_softmax.GetOutput(), y);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}