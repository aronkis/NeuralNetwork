#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>

class Loss
{
public:
    Loss() = default;
    virtual ~Loss() = default;

    void calculateLoss(const Eigen::MatrixXd& predictions, 
                       const Eigen::MatrixXi& targets);

    double GetLoss() const;

protected:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& predictions, 
                                    const Eigen::MatrixXi& targets) = 0;

private:
    double loss_ = 0.0;
};

#endif // __LOSS_H__