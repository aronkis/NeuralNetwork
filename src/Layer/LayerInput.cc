#include "LayerInput.h"

void NEURAL_NETWORK::LayerInput::forward(const Eigen::MatrixXd& inputs, bool training) 
{
    (void)training; // unused parameter
    output_ = inputs;
}

void NEURAL_NETWORK::LayerInput::backward(const Eigen::MatrixXd& dvalues) 
{
    (void)dvalues; // unused parameter
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerInput::GetOutput() const 
{
    return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::LayerInput::GetDInput() const 
{
    static const Eigen::MatrixXd empty_matrix;
    return empty_matrix;
}

Eigen::MatrixXd NEURAL_NETWORK::LayerInput::predictions() const 
{
    return output_;
}