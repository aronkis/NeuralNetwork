#include <Eigen/Dense>

class Helpers
{
public:
    Helpers() = delete;
    ~Helpers() = delete;
    static void ReadSpiralIntoEigen(const std::string& filename,
                             Eigen::MatrixXd& coordinates,
                             Eigen::RowVectorXi& classes);
};

