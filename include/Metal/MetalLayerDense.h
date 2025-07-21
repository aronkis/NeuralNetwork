#ifndef __METAL_LAYER_DENSE_H__
#define __METAL_LAYER_DENSE_H__

#include "LayerDense.h"
#include "MetalDevice.h"

namespace NEURAL_NETWORK {
namespace Metal {

class MetalLayerDense : public LayerDense {
public:
    MetalLayerDense(int n_inputs, int n_neurons, bool useGPU = true);
    MetalLayerDense(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases, bool useGPU = true);
    
    void forward(const Eigen::MatrixXd& inputs) override;
    void backward(const Eigen::MatrixXd& d_values) override;
    
    void setUseGPU(bool useGPU) { useGPU_ = useGPU; }
    bool isUsingGPU() const { return useGPU_ && MetalDevice::getInstance().isMetalAvailable(); }
    
private:
    bool useGPU_;
};

} // namespace Metal
} // namespace NEURAL_NETWORK

#endif // __METAL_LAYER_DENSE_H__
