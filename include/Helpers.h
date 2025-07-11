#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <Eigen/Dense>

class Helpers
{
public:
    Helpers() = delete;
    ~Helpers() = delete;
    
    static void ReadSpiralIntoEigen(const std::string& filename,
                                    Eigen::MatrixXd& coordinates,
                                    Eigen::MatrixXi& classes);

    static double CalculateAccuracy(const Eigen::MatrixXd& output, 
                                          Eigen::MatrixXi  targets);
};

#endif // __HELPERS_H__