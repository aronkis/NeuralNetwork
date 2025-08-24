#include <iostream>
#include "Model.h"
#include "ActivationReLU.h"
#include "ActivationSoftmax.h"
#include "LossCategoricalCrossentropy.h"
#include "LossMeanSquaredError.h"
#include "Adam.h"
#include "Helpers.h"
#include "AccuracyCategorical.h"
#include "LayerDropout.h"
#include "ActivationLinear.h"
#include "AccuracyRegression.h"
#include "ActivationSigmoid.h"
#include "LossBinaryCrossEntropy.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"

#ifndef NN_EPOCHS
#define NN_EPOCHS 10000
#endif
#ifndef NN_PRINT_EVERY
#define NN_PRINT_EVERY 1000
#endif

// #define CLASSIFICATION
#define BINARY_CLASSIFICATION
// #define REGRESSION

#ifdef REGRESSION
int main()
{
    NEURAL_NETWORK::Model model;

    model.add(std::make_unique<NEURAL_NETWORK::LayerInput>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(1, 64));
    model.add(std::make_unique<NEURAL_NETWORK::ActivationReLU>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(64, 64)); 
    model.add(std::make_unique<NEURAL_NETWORK::ActivationReLU>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model.add(std::make_unique<NEURAL_NETWORK::ActivationLinear>());

    model.set(
        std::make_unique<NEURAL_NETWORK::LossMeanSquaredError>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.005, 1e-3, 0.9, 0.999, 1e-7),
        std::make_unique<NEURAL_NETWORK::AccuracyRegression>()
    );

    model.finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "../data/sine.txt";
    std::string validation_filename = "../data/sine_validation.txt";
    NEURAL_NETWORK::Helpers::Read1DIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::Read1DIntoEigen(validation_filename, X_test, y_test);

    model.train(X_train, y_train, NN_EPOCHS, NN_PRINT_EVERY, std::make_pair(X_test, y_test), true);

}
#endif

#ifdef BINARY_CLASSIFICATION
int main()
{
     NEURAL_NETWORK::Model model;

    model.add(std::make_unique<NEURAL_NETWORK::LayerInput>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(2, 64, 0, 5e-4, 0, 5e-4));
    model.add(std::make_unique<NEURAL_NETWORK::ActivationReLU>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(64, 1)); 
    model.add(std::make_unique<NEURAL_NETWORK::ActivationSigmoid>());

    model.set(
        std::make_unique<NEURAL_NETWORK::LossBinaryCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.019, 5e-7, 0.9, 0.999, 1e-7),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>()
    );

    model.finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "../data/binary.txt";
    std::string validation_filename = "../data/binary_validation.txt";
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

    model.train(X_train, y_train, NN_EPOCHS, NN_PRINT_EVERY, std::make_pair(X_test, y_test), false);

}
#endif

#ifdef CLASSIFICATION
int main() {
    NEURAL_NETWORK::Model model;

    model.add(std::make_unique<NEURAL_NETWORK::LayerInput>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(2, 512, 0.0, 5e-4, 0.0, 5e-4));
    model.add(std::make_unique<NEURAL_NETWORK::ActivationReLU>());
    model.add(std::make_unique<NEURAL_NETWORK::LayerDropout>(0.1));
    model.add(std::make_unique<NEURAL_NETWORK::LayerDense>(512, 3)); 
    model.add(std::make_unique<NEURAL_NETWORK::ActivationSoftmaxLossCategoricalCrossEntropy>());

    model.set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.05, 5e-5),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>()
    );

    model.finalize();

    Eigen::MatrixXd X_train; 
    Eigen::MatrixXd y_train; 
    Eigen::MatrixXd X_test;  
    Eigen::MatrixXd y_test;  

    std::string filename = "../data/points.txt";
    std::string validation_filename = "../data/validation.txt";
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(filename, X_train, y_train);
    NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(validation_filename, X_test, y_test);

    model.train(X_train, y_train, NN_EPOCHS, NN_PRINT_EVERY, std::make_pair(X_test, y_test), false);

    return 0;
}
#endif