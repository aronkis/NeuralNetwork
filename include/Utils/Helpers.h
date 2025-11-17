#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <Eigen/Dense>
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

		// General CSV loader for matrices with many features
		void ReadCSVMatrix(const std::string& filename,
						   Eigen::MatrixXd& matrix,
						   char delimiter = ',');

		// Load integer labels from CSV (one label per line)
		void ReadCSVLabels(const std::string& filename,
						   Eigen::VectorXi& labels);

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
					  Eigen::MatrixXd& X,
					  Eigen::MatrixXd& y);

		void CreateDataSets(const std::string& dataset_url,
							const std::string& output_dir,
							Eigen::MatrixXd& X,
							Eigen::MatrixXd& y,
							Eigen::MatrixXd& X_test,
							Eigen::MatrixXd& y_test);

		void ShuffleData(Eigen::MatrixXd& X,
						 Eigen::MatrixXd& y);

		void ScaleData(Eigen::MatrixXd& X);

		void ReadSingleImage(const std::string& filename,
							 Eigen::MatrixXd& image);

		// Flatten spatial data for transition from CNN to Dense layers
		Eigen::MatrixXd Flatten(const Eigen::MatrixXd& spatial_data);
	};
} // namespace NEURAL_NETWORK

#endif // __HELPERS_H__