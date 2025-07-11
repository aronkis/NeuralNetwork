#include "LossCategoricalCrossentropy.h"

Eigen::MatrixXd LossCategoricalCrossentropy::forward(const Eigen::MatrixXd& predictions, 
                                                      const Eigen::MatrixXi& targets)
{
    int samples = predictions.rows();

    Eigen::MatrixXd y_pred_clipped = predictions.array().max(1e-7).min(1-1e-7);
    Eigen::VectorXd correct_confidences(samples);

    if (targets.cols() == 1) 
    {
        for (int i = 0; i < samples; ++i) 
        {
            correct_confidences(i) = y_pred_clipped(i, targets(i, 0));
        }
    } 
    else if(targets.cols() == 2)
    {
        correct_confidences = (y_pred_clipped.array() * (targets.cast<double>()).array()).rowwise().sum();
    }

    return -correct_confidences.array().log();
}