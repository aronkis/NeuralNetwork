#ifndef __METAL_ACTIVATION_RELU_H__
#define __METAL_ACTIVATION_RELU_H__

#include "ActivationReLU.h"
#include "MetalDevice.h"

namespace NEURAL_NETWORK {
namespace Metal {

class MetalActivationReLU : public ActivationReLU {
public:
    MetalActivationReLU(bool useGPU = true);
    
    void forward(const Eigen::MatrixXd& inputs) override;
    void backward(const Eigen::MatrixXd& dvalues) override;
    
    void setUseGPU(bool useGPU) { useGPU_ = useGPU; }
    bool isUsingGPU() const { return useGPU_ && MetalDevice::getInstance().isMetalAvailable(); }
    
private:
    bool useGPU_;
};

} // namespace Metal
} // namespace NEURAL_NETWORK

#endif // __METAL_ACTIVATION_RELU_H__
