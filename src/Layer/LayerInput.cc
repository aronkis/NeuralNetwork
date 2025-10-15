#include "LayerInput.h"

void NEURAL_NETWORK::LayerInput::forward(const Eigen::Tensor<double, 2>& inputs,
										 bool training)
{
    output_ = inputs;
}

void NEURAL_NETWORK::LayerInput::backward(const Eigen::Tensor<double, 2>& dvalues)
{
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerInput::GetOutput() const
{
    return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::LayerInput::GetDInput() const
{
    static const Eigen::Tensor<double, 2> empty_tensor;
    return empty_tensor;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::LayerInput::predictions() const
{
    return output_;
}