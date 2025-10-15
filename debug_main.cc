#include "Model.h"
#include <iostream>

int main()
{
    std::cout << "Creating small test tensors..." << std::endl;

    // Create a very small test case
    Eigen::Tensor<double, 4> X_tensor(2, 3, 3, 1);  // 2 samples, 3x3 images, 1 channel
    Eigen::Tensor<double, 2> y_tensor(2, 1);        // 2 labels

    // Initialize with simple data
    for (int i = 0; i < 2; i++)
    {
        for (int h = 0; h < 3; h++)
        {
            for (int w = 0; w < 3; w++)
            {
                X_tensor(i, h, w, 0) = (i + 1) * 0.1 + h * 0.01 + w * 0.001;
            }
        }
        y_tensor(i, 0) = i;
    }

    std::cout << "Creating simple model..." << std::endl;

    NEURAL_NETWORK::Model model;
    model.Add(std::make_shared<NEURAL_NETWORK::LayerInput>());

    // Simple dense layer only - no CNN layers to isolate the issue
    model.Add(std::make_shared<NEURAL_NETWORK::LayerDense>(9, 2));  // 3x3x1 = 9 inputs, 2 outputs
    model.Add(std::make_shared<NEURAL_NETWORK::ActivationSoftmax>());

    model.Set(
        std::make_unique<NEURAL_NETWORK::LossCategoricalCrossEntropy>(),
        std::make_unique<NEURAL_NETWORK::AccuracyCategorical>(),
        std::make_unique<NEURAL_NETWORK::Adam>(0.001, 1e-7)
    );

    std::cout << "Finalizing model..." << std::endl;
    model.Finalize();

    std::cout << "Starting training..." << std::endl;
    model.Train(X_tensor, y_tensor, 1, 1, 1, X_tensor, y_tensor);

    std::cout << "Training completed successfully!" << std::endl;
    return 0;
}