#include "LossBinaryCrossEntropy.h"

void NEURAL_NETWORK::LossBinaryCrossEntropy::forward(const Eigen::MatrixXd& predictions, 
													 const Eigen::MatrixXi& targets)
{
    Eigen::MatrixXd y_pred_clipped = predictions.array().cwiseMax(1e-7).cwiseMin(1.0 - 1e-7);

    output_ = -(targets.cast<double>().array() * y_pred_clipped.array().log() + 
			   (1.0 - targets.cast<double>().array()) * (1.0 - y_pred_clipped.array()).log());

    output_ = output_.rowwise().mean();
}

void NEURAL_NETWORK::LossBinaryCrossEntropy::backward(const Eigen::MatrixXd& d_values, 
													  const Eigen::MatrixXi& targets)
{
	int samples = d_values.rows();
	int outputs = d_values.cols();

    Eigen::MatrixXd d_values_clipped = d_values.array().cwiseMax(1e-7).cwiseMin(1.0 - 1e-7);

	d_inputs_ = -(targets.cast<double>().array() / d_values_clipped.array() - 
				 (1.0 - targets.cast<double>().array()) / (1.0 - d_values_clipped.array())) / outputs;

	d_inputs_ /= samples;
}