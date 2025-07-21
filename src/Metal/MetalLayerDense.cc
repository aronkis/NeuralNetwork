#include "MetalLayerDense.h"

namespace NEURAL_NETWORK {
namespace Metal {

MetalLayerDense::MetalLayerDense(int n_inputs, int n_neurons, bool useGPU) 
    : LayerDense(n_inputs, n_neurons), useGPU_(useGPU) {
}

MetalLayerDense::MetalLayerDense(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases, bool useGPU)
    : LayerDense(weights, biases), useGPU_(useGPU) {
}

void MetalLayerDense::forward(const Eigen::MatrixXd& inputs) {
    inputs_ = inputs;
    
    if (isUsingGPU()) {
        // Use Metal for matrix multiplication
        Eigen::MatrixXd temp;
        MetalDevice::getInstance().matrixMultiply(inputs, weights_, temp);
        
        // Add biases (CPU for now, as it's a simple broadcast operation)
        output_ = temp.rowwise() + biases_;
    } else {
        // Fallback to CPU implementation
        LayerDense::forward(inputs);
    }
}

void MetalLayerDense::backward(const Eigen::MatrixXd& d_values) {
    if (isUsingGPU()) {
        // Use Metal for matrix operations where beneficial
        MetalDevice& metalDevice = MetalDevice::getInstance();
        
        // d_weights = inputs_.transpose() * d_values
        Eigen::MatrixXd inputs_transposed = inputs_.transpose();
        metalDevice.matrixMultiply(inputs_transposed, d_values, d_weights_);
        
        // d_biases = d_values.colwise().sum() (CPU for now)
        d_biases_ = d_values.colwise().sum();
        
        // d_inputs = d_values * weights_.transpose()
        Eigen::MatrixXd weights_transposed = weights_.transpose();
        metalDevice.matrixMultiply(d_values, weights_transposed, d_inputs_);
    } else {
        // Fallback to CPU implementation
        LayerDense::backward(d_values);
    }
}

} // namespace Metal
} // namespace NEURAL_NETWORK
