#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>

namespace NEURAL_NETWORK 
{
	namespace Serialization 
	{
		void WriteRaw(std::ofstream& stream, 
					  const void* data, 
					  size_t size);
		
		void ReadRaw(std::ifstream& stream, 
					 void* data, 
					 size_t size);
		
		void WriteString(std::ofstream& stream, 
						 const std::string& str);
		std::string ReadString(std::ifstream& stream);

		void WriteVectorDouble(std::ofstream& stream, const 
							   std::vector<double>& vec);
		std::vector<double> ReadVectorDouble(std::ifstream& stream);

		void WriteMatrix(std::ofstream& stream, 
						 const Eigen::MatrixXd& mat);
		Eigen::MatrixXd ReadMatrix(std::ifstream& stream);

		void WriteRowVector(std::ofstream& stream, 
							const Eigen::RowVectorXd& vec);
		Eigen::RowVectorXd ReadRowVector(std::ifstream& stream);
	} // namespace Serialization
} // namespace NEURAL_NETWORK

#endif // __SERIALIZATION_H__