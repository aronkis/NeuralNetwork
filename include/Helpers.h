#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
	namespace Helpers
	{
		void ReadSpiralIntoEigen(const std::string& filename,
								 Eigen::MatrixXd& coordinates,
								 Eigen::MatrixXi& classes);

		double CalculateAccuracy(const Eigen::MatrixXd& output,
								 Eigen::MatrixXi& targets);

		Eigen::MatrixXd MatrixSquare(const Eigen::MatrixXd& matrix);
		Eigen::ArrayXXd MatrixSquareRootToArray(const Eigen::MatrixXd& matrix);
	};

} // namespace NEURAL_NETWORK

#endif // __HELPERS_H__