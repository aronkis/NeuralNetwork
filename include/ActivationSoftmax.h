#ifndef __ACTIVATION_SOFTMAX_H__
#define __ACTIVATION_SOFTMAX_H__

#include <Eigen/Dense>

class ActivationSoftmax
{
public:
    ActivationSoftmax() = default;
    ~ActivationSoftmax() = default;

    void forward(const Eigen::MatrixXd inputs);
    const Eigen::MatrixXd& GetOutput() const;

private:
    Eigen::MatrixXd output_;

};

#endif // __ACTIVATION_SOFTMAX_H__