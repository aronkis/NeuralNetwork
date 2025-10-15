#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace NEURAL_NETWORK
{
	namespace Helpers
	{
		void ReadSpiralIntoEigen(const std::string& filename,
								 Eigen::MatrixXd& coordinates,
								 Eigen::MatrixXd& classes);

		void Read1DIntoEigen(const std::string& filename,
							 Eigen::MatrixXd& input,
							 Eigen::MatrixXd& output);

		void ReadFromCSVIntoEigen(const std::string & filename,
								  Eigen::MatrixXd& input,
								  Eigen::MatrixXd& output,
								  char delimiter = ',');

		void DownloadData(const std::string url,
						  const std::string output_dir,
						  const std::string filename);
		
		void UnzipFile(const std::string& directory,
				   const std::string& filename,
				   const std::string& target);

		void FetchData(const std::string url,
					   const std::string output_path,
					   const std::string filename,
					   const bool unzip = false);

		void LoadData(const std::string& path,
					  Eigen::Tensor<double, 4>& X_tensor,
					  Eigen::Tensor<double, 2>& y_tensor);

		void CreateDataSets(const std::string& dataset_url,
							const std::string& output_dir,
							Eigen::Tensor<double, 4>& X_tensor,
							Eigen::Tensor<double, 2>& y_tensor,
							Eigen::Tensor<double, 4>& X_test_tensor,
							Eigen::Tensor<double, 2>& y_test_tensor);

		void ShuffleData(Eigen::Tensor<double, 4>& X_tensor,
						 Eigen::Tensor<double, 2>& y_tensor);

		void ScaleData(Eigen::Tensor<double, 4>& X_tensor);

		void ReadSingleImage(const std::string& filename,
							 Eigen::Tensor<double, 4>& image_tensor);

		// Flatten spatial data for transition from CNN to Dense layers
		Eigen::MatrixXd Flatten(const Eigen::MatrixXd& spatial_data);

		// Convert 4D tensor to 2D tensor (flattening spatial dimensions)
		Eigen::Tensor<double, 2> TensorToTensor2D(const Eigen::Tensor<double, 4>& tensor);

		// Convert matrix to 2D tensor
		Eigen::Tensor<double, 2> MatrixToTensor2D(const Eigen::MatrixXd& matrix);

		// Convert RowVector to 1D tensor
		Eigen::Tensor<double, 1> RowVectorToTensor1D(const Eigen::RowVectorXd& rowvec);

		// Convert 2D tensor to matrix (for backward compatibility)
		Eigen::MatrixXd TensorToMatrix(const Eigen::Tensor<double, 2>& tensor);

		// Convert 1D tensor to RowVector (for backward compatibility)
		Eigen::RowVectorXd TensorToRowVector(const Eigen::Tensor<double, 1>& tensor);
	};
} // namespace NEURAL_NETWORK

#endif // __HELPERS_H__