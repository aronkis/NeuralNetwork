#ifndef __LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "Loss.h"

class LossCategoricalCrossentropy : public Loss
{
protected:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& predictions, 
                            const Eigen::MatrixXi& targets) override;
};

#endif // __LOSS_CATEGORICAL_CROSSENTROPY_H__