#include "Helpers.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

int main()
{
    std::cout << "Testing Unified Helper Functions\n";
    std::cout << "=================================\n\n";

    std::string dataset_url = "https://nnfs.io/datasets/fashion_mnist_images.zip";
    std::string output_dir = "data/";

    // Test with Matrix type (flattened data)
    std::cout << "1. Testing with Eigen::MatrixXd (flattened data):\n";
    {
        Eigen::MatrixXd X_matrix, X_test_matrix, y_matrix, y_test_matrix;

        // This should call the matrix-based implementation
        NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir,
                                                X_matrix, y_matrix,
                                                X_test_matrix, y_test_matrix);

        std::cout << "   Matrix data loaded successfully!\n";
        std::cout << "   Training data shape: " << X_matrix.rows() << " x " << X_matrix.cols() << "\n";
        std::cout << "   Test data shape: " << X_test_matrix.rows() << " x " << X_test_matrix.cols() << "\n";
    }

    std::cout << "\n2. Testing with Eigen::Tensor<double, 4> (spatial data):\n";
    {
        Eigen::Tensor<double, 4> X_tensor, X_test_tensor;
        Eigen::MatrixXd y_tensor, y_test_tensor;

        // This should call the tensor-based implementation
        NEURAL_NETWORK::Helpers::CreateDataSets(dataset_url, output_dir,
                                                X_tensor, y_tensor,
                                                X_test_tensor, y_test_tensor);

        std::cout << "   Tensor data loaded successfully!\n";
        std::cout << "   Training tensor dimensions: "
                  << X_tensor.dimension(0) << " x " << X_tensor.dimension(1)
                  << " x " << X_tensor.dimension(2) << " x " << X_tensor.dimension(3) << "\n";
        std::cout << "   Test tensor dimensions: "
                  << X_test_tensor.dimension(0) << " x " << X_test_tensor.dimension(1)
                  << " x " << X_test_tensor.dimension(2) << " x " << X_test_tensor.dimension(3) << "\n";
    }

    std::cout << "\n3. Demonstrating the unified interface:\n";
    std::cout << "   Both calls used the same function name 'CreateDataSets'\n";
    std::cout << "   Compiler automatically selected the correct implementation\n";
    std::cout << "   based on the argument types!\n\n";

    std::cout << "Test completed successfully! 🎉\n";
    std::cout << "The unified interface works for both tensors and matrices.\n";

    return 0;
}