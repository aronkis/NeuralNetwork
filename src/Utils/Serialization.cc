#include "Serialization.h"

void NEURAL_NETWORK::Serialization::WriteRaw(std::ofstream& stream, 
											 const void* data, size_t size) 
{
	stream.write(reinterpret_cast<const char*>(data), size);
}

void NEURAL_NETWORK::Serialization::ReadRaw(std::ifstream& stream, 
											void* data, 
											size_t size) 
{
	stream.read(reinterpret_cast<char*>(data), size);
}

void NEURAL_NETWORK::Serialization::WriteString(std::ofstream& stream, 
												const std::string& str) 
{
	size_t len = str.size();
	stream.write(reinterpret_cast<const char*>(&len), sizeof(len));
	stream.write(str.data(), len);
}

std::string NEURAL_NETWORK::Serialization::ReadString(std::ifstream& stream) 
{
	size_t len;
	stream.read(reinterpret_cast<char*>(&len), sizeof(len));

	std::string str(len, '\0');
	stream.read(&str[0], len);

	return str;
}

void NEURAL_NETWORK::Serialization::WriteVectorDouble(std::ofstream& stream, 
													  const std::vector<double>& vec) 
{
	size_t len = vec.size();
	stream.write(reinterpret_cast<const char*>(&len), sizeof(len));
	stream.write(reinterpret_cast<const char*>(vec.data()), 
				 len * sizeof(double));
}

std::vector<double> NEURAL_NETWORK::Serialization::ReadVectorDouble(std::ifstream& stream) 
{
	size_t len;
	stream.read(reinterpret_cast<char*>(&len), sizeof(len));

	std::vector<double> vec(len);
	stream.read(reinterpret_cast<char*>(vec.data()), len * sizeof(double));

	return vec;
}

void NEURAL_NETWORK::Serialization::WriteMatrix(std::ofstream& stream, 
												const Eigen::MatrixXd& mat) 
{
	int rows = mat.rows();
	int cols = mat.cols();

	stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
	stream.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
	stream.write(reinterpret_cast<const char*>(mat.data()), 
				 rows * cols * sizeof(double));
}

Eigen::MatrixXd NEURAL_NETWORK::Serialization::ReadMatrix(std::ifstream& stream) 
{
	int rows;
	int cols;

	stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));

	Eigen::MatrixXd mat(rows, cols);
	stream.read(reinterpret_cast<char*>(mat.data()), 
				rows * cols * sizeof(double));

	return mat;
}

void NEURAL_NETWORK::Serialization::WriteRowVector(std::ofstream& stream, 
												   const Eigen::RowVectorXd& vec) 
{
	int cols = vec.cols();

	stream.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
	stream.write(reinterpret_cast<const char*>(vec.data()), 
				 cols * sizeof(double));
}

Eigen::RowVectorXd NEURAL_NETWORK::Serialization::ReadRowVector(std::ifstream& stream) 
{
	int cols;
	stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));

	Eigen::RowVectorXd vec(cols);
	stream.read(reinterpret_cast<char*>(vec.data()), cols * sizeof(double));
	
	return vec;
}