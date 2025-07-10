#include "ActivationSoftmax.h"
#include <iostream>


void ActivationSoftmax::forward(const Eigen::MatrixXd inputs)
{
    Eigen::MatrixXd exp_values = (inputs.colwise() - (inputs.rowwise().maxCoeff())).array().exp();

    output_ = exp_values.array().colwise() / exp_values.rowwise().sum().array();
}

const Eigen::MatrixXd& ActivationSoftmax::GetOutput() const
{
    return output_;
}