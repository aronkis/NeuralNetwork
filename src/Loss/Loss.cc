#include "Loss.h"
#include <iostream>

void Loss::calculateLoss(const Eigen::MatrixXd& predictions, const Eigen::MatrixXi& targets)
{
    Eigen::MatrixXd sample_loss = forward(predictions, targets);
    loss_ = sample_loss.array().mean();
}

double Loss::GetLoss() const
{
    return loss_;
}   