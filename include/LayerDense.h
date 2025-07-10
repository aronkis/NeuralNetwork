#ifndef __LAYERDENSE_H__
#define __LAYERDENSE_H__

#include <Eigen/Dense>

class LayerDense
{

public:
    LayerDense(int n_inputs, int n_neurons);
    ~LayerDense() = default;

    void forward(const Eigen::MatrixXd inputs);
    const Eigen::MatrixXd& GetOutput() const;

private:
    Eigen::MatrixXd weights_;
    Eigen::RowVectorXd biases_;
    Eigen::MatrixXd output_;
};

#endif // __LAYERDENSE_H__