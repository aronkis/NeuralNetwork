#include "MetalActivationReLU.h"

namespace NEURAL_NETWORK {
namespace Metal {

MetalActivationReLU::MetalActivationReLU(bool useGPU) : useGPU_(useGPU) {
}

void MetalActivationReLU::forward(const Eigen::MatrixXd& inputs) {
    inputs_ = inputs;
    
    if (isUsingGPU()) {
        MetalDevice::getInstance().relu(inputs, output_);
    } else {
        // Fallback to CPU implementation
        ActivationReLU::forward(inputs);
    }
}

void MetalActivationReLU::backward(const Eigen::MatrixXd& dvalues) {
    if (isUsingGPU()) {
        MetalDevice::getInstance().reluBackward(dvalues, inputs_, dinput_);
    } else {
        // Fallback to CPU implementation
        ActivationReLU::backward(dvalues);
    }
}

} // namespace Metal
} // namespace NEURAL_NETWORK
