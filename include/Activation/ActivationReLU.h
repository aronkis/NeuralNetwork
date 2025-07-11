#ifndef __ACTIVATION_RELU_H__
#define __ACTIVATION_RELU_H__

#include <Eigen/Dense>

class ActivationReLU
{
public:
    ActivationReLU() = default;
    ~ActivationReLU() = default;

    void forward(const Eigen::MatrixXd inputs);
    const Eigen::MatrixXd& GetOutput() const;

private:
    Eigen::MatrixXd output_;
};

#endif //__ACTIVATION_RELU_H__