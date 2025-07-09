#include <Eigen/Dense>

class LayerDense
{

public:
    LayerDense(int n_inputs, int n_neurons);
    ~LayerDense() = default;

    void forward(Eigen::MatrixXd inputs);
    Eigen::MatrixXd GetOutput();

private:
    Eigen::MatrixXd weights;
    Eigen::RowVectorXd biases;
    Eigen::MatrixXd output;
};

