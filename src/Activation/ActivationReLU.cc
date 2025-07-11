#include "ActivationReLU.h"

void ActivationReLU::forward(const Eigen::MatrixXd inputs)
{
    output_ = inputs.cwiseMax(0.0);
}

const Eigen::MatrixXd& ActivationReLU::GetOutput() const
{
    return output_;
}