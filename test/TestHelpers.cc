#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>
#include "Helpers.h"

class HelpersTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		temp_dir = "/tmp/neural_network_test/";
		std::filesystem::create_directories(temp_dir);

		test_csv_file = temp_dir + "test_data.csv";
		test_1d_file = temp_dir + "test_1d.txt";
		test_spiral_file = temp_dir + "test_spiral.txt";

		CreateTestFiles();
	}

	static void WriteBMP(const std::string& path,
						int width,
						int height,
						const std::vector<unsigned char>& rgba)
	{
		const int bytes_per_pixel = 3;
		const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
		const int pixel_data_size = row_stride * height;
		const int file_size = 14 + 40 + pixel_data_size;

		std::ofstream out(path, std::ios::binary);
		if (!out.is_open())
		{
			throw std::runtime_error("Failed to open BMP file for writing");
		}

		out.put('B');
		out.put('M');
		out.write(reinterpret_cast<const char*>(&file_size), 4);
		int reserved = 0;
		out.write(reinterpret_cast<const char*>(&reserved), 4);
		int offset = 14 + 40;
		out.write(reinterpret_cast<const char*>(&offset), 4);

		int dib_header_size = 40;
		out.write(reinterpret_cast<const char*>(&dib_header_size), 4);
		out.write(reinterpret_cast<const char*>(&width), 4);
		out.write(reinterpret_cast<const char*>(&height), 4);
		short planes = 1;
		short bpp = bytes_per_pixel * 8;
		out.write(reinterpret_cast<const char*>(&planes), 2);
		out.write(reinterpret_cast<const char*>(&bpp), 2);
		int compression = 0;
		out.write(reinterpret_cast<const char*>(&compression), 4);
		out.write(reinterpret_cast<const char*>(&pixel_data_size), 4);
		int ppm = 2835; 
		out.write(reinterpret_cast<const char*>(&ppm), 4);
		out.write(reinterpret_cast<const char*>(&ppm), 4);
		int palette_colors = 0;
		int important_colors = 0;
		out.write(reinterpret_cast<const char*>(&palette_colors), 4);
		out.write(reinterpret_cast<const char*>(&important_colors), 4);

		std::vector<unsigned char> row(row_stride, 0);
		for (int y = height - 1; y >= 0; --y)
		{
			std::fill(row.begin(), row.end(), 0);
			for (int x = 0; x < width; x++)
			{
				const size_t rgba_idx = (y * width + x) * 4;
				const size_t bmp_idx = x * bytes_per_pixel;
				if (rgba_idx + 3 > rgba.size())
				{
					throw std::out_of_range("WriteBMP received insufficient RGBA data for requested image size");
				}
				if (bmp_idx + bytes_per_pixel > row.size())
				{
					throw std::out_of_range("WriteBMP computed row stride smaller than required for pixel data");
				}
				row[bmp_idx + 0] = rgba[rgba_idx + 2]; 
				row[bmp_idx + 1] = rgba[rgba_idx + 1]; 
				row[bmp_idx + 2] = rgba[rgba_idx + 0]; 
			}
			out.write(reinterpret_cast<const char*>(row.data()), row_stride);
		}
	}

	void TearDown() override 
	{
		if (std::filesystem::exists(temp_dir)) 
		{
			std::filesystem::remove_all(temp_dir);
		}
	}

	void CreateTestFiles() 
	{
		CreateCSVTestFile();

		Create1DTestFile();

		CreateSpiralTestFile();
	}

	void CreateCSVTestFile()
	{
		std::ofstream csv_file(test_csv_file);
		csv_file << "1.0,  2.0,  10.0,  0\n";
		csv_file << "4.0,  5.0,  20.0,  1\n";
		csv_file << "7.0,  8.0,  30.0,  2\n";
		csv_file << "10.0, 11.0, 40.0, 1\n";
		csv_file.close();
	}

	void Create1DTestFile() 
	{
		std::ofstream file_1d(test_1d_file);
		file_1d << "1.0 10.0\n";
		file_1d << "2.0 20.0\n";
		file_1d << "3.0 30.0\n";
		file_1d << "4.0 40.0\n";
		file_1d.close();
	}

	void CreateSpiralTestFile() 
	{
		std::ofstream spiral_file(test_spiral_file);
		spiral_file << "1.0 2.0 0\n";
		spiral_file << "3.0 4.0 1\n";
		spiral_file << "5.0 6.0 0\n";
		spiral_file << "7.0 8.0 1\n";
		spiral_file.close();
	}

	std::string temp_dir;
	std::string test_csv_file;
	std::string test_1d_file;
	std::string test_spiral_file;
	const double tolerance = 1e-10;
};

TEST_F(HelpersTest, ShuffleDataBasic) 
{
	Eigen::MatrixXd X(4, 2);
	X << 1, 2,
		 3, 4,
		 5, 6,
		 7, 8;

	Eigen::MatrixXd y(4, 1);
	y << 10, 20, 30, 40;

	Eigen::MatrixXd X_original = X;
	Eigen::MatrixXd y_original = y;

	NEURAL_NETWORK::Helpers::ShuffleData(X, y);

	EXPECT_EQ(X.rows(), X_original.rows());
	EXPECT_EQ(X.cols(), X_original.cols());
	EXPECT_EQ(y.rows(), y_original.rows());
	EXPECT_EQ(y.cols(), y_original.cols());

	std::vector<double> original_x_sum, shuffled_x_sum;
	std::vector<double> original_y_values, shuffled_y_values;

	for (int i = 0; i < X_original.rows(); i++) 
	{
		original_x_sum.push_back(X_original.row(i).sum());
		shuffled_x_sum.push_back(X.row(i).sum());
		original_y_values.push_back(y_original(i, 0));
		shuffled_y_values.push_back(y(i, 0));
	}

	std::sort(original_x_sum.begin(), original_x_sum.end());
	std::sort(shuffled_x_sum.begin(), shuffled_x_sum.end());
	std::sort(original_y_values.begin(), original_y_values.end());
	std::sort(shuffled_y_values.begin(), shuffled_y_values.end());

	for (size_t i = 0; i < original_x_sum.size(); i++) 
	{
		EXPECT_NEAR(original_x_sum[i], shuffled_x_sum[i], tolerance);
		EXPECT_NEAR(original_y_values[i], shuffled_y_values[i], tolerance);
	}
}

TEST_F(HelpersTest, ShuffleDataPreservesCorrespondence) 
{
	
	Eigen::MatrixXd X(3, 1);
	X << 1, 2, 3;

	Eigen::MatrixXd y(3, 1);
	y << 10, 20, 30;  

	NEURAL_NETWORK::Helpers::ShuffleData(X, y);

	for (int i = 0; i < X.rows(); i++) 
	{
		EXPECT_NEAR(y(i, 0), X(i, 0) * 10, tolerance);
	}
}

TEST_F(HelpersTest, ScaleDataBasic) 
{
	Eigen::MatrixXd X(4, 2);
	X << 1, 10,
		 2, 20,
		 3, 30,
		 4, 40;

	Eigen::MatrixXd X_original = X;

	NEURAL_NETWORK::Helpers::ScaleData(X);

	EXPECT_FALSE(X.isApprox(X_original, tolerance));

	for (int col = 0; col < X.cols(); col++)
	{
		Eigen::VectorXd column = X.col(col);
		double mean = column.mean();
		double std_dev = std::sqrt((column.array() - mean).square().mean());

		EXPECT_LT(std::abs(mean), 1.0);
		EXPECT_GT(std_dev, 0.0);
	}
}

TEST_F(HelpersTest, ScaleDataSingleColumn) 
{
	Eigen::MatrixXd X(4, 1);
	X << 5, 10, 15, 20;

	NEURAL_NETWORK::Helpers::ScaleData(X);

	double mean = X.col(0).mean();
	double std_dev = std::sqrt((X.col(0).array() - mean).square().mean());

	EXPECT_NEAR(mean, 0.0, 1e-10);
	EXPECT_NEAR(std_dev, 1.0, 1e-10);
}

TEST_F(HelpersTest, ScaleDataConstantValues) 
{
	Eigen::MatrixXd X(4, 1);
	X << 5, 5, 5, 5;

	EXPECT_NO_THROW(NEURAL_NETWORK::Helpers::ScaleData(X));

	for (int i = 1; i < X.rows(); i++) 
	{
		EXPECT_DOUBLE_EQ(X(i, 0), X(0, 0));
	}
}

class HelpersDataCreationTest : public HelpersTest 
{};

TEST_F(HelpersDataCreationTest, ReadSingleImageHandling) 
{
	std::string test_image = temp_dir + "tiny.bmp";
	std::vector<unsigned char> rgba = {
		255,   0,   0, 255,   0, 255,   0, 255,
		  0,   0, 255, 255, 255, 255,   0, 255
	};
	WriteBMP(test_image, 2, 2, rgba);

	Eigen::MatrixXd image;
	NEURAL_NETWORK::Helpers::ReadSingleImage(test_image, image);

	EXPECT_EQ(image.rows(), 1);
	EXPECT_EQ(image.cols(), 4);
	EXPECT_NEAR(image(0, 0), 255.0, tolerance);
	EXPECT_NEAR(image(0, 1),   0.0, tolerance);
	EXPECT_NEAR(image(0, 2),   0.0, tolerance);
	EXPECT_NEAR(image(0, 3), 255.0, tolerance);
}

TEST_F(HelpersTest, ScalingPerformanceReasonable) 
{
	int n_samples = 1000;
	int n_features = 100;

	Eigen::MatrixXd large_data = Eigen::MatrixXd::Random(n_samples, n_features);

	auto start_time = std::chrono::high_resolution_clock::now();
	NEURAL_NETWORK::Helpers::ScaleData(large_data);
	auto end_time = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	EXPECT_LT(duration.count(), 1000);
}
