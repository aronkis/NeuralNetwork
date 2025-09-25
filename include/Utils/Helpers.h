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
	};
} // namespace NEURAL_NETWORK

#endif // __HELPERS_H__