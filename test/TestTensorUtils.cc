#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "TensorUtils.h"

class TensorUtilsTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		tolerance = 1e-10;
	}
	
	double tolerance;
};

TEST_F(TensorUtilsTest, MatrixToTensor4DBasic)
{
	Eigen::MatrixXd matrix(2, 8); 
	matrix << 1, 2, 3, 4, 5, 6, 7, 8,
			  9, 10, 11, 12, 13, 14, 15, 16;
	
	Eigen::Tensor<double, 4> tensor(2, 2, 2, 2);
	
	EXPECT_NO_THROW(NEURAL_NETWORK::TensorUtils::MatrixToTensor4D(
		matrix, tensor, 2, 2, 2, 2));
	
	EXPECT_EQ(tensor.dimension(0), 2); 
	EXPECT_EQ(tensor.dimension(1), 2); 
	EXPECT_EQ(tensor.dimension(2), 2); 
	EXPECT_EQ(tensor.dimension(3), 2); 
}

TEST_F(TensorUtilsTest, Tensor4DToMatrixBasic)
{
	Eigen::Tensor<double, 4> tensor(1, 2, 2, 1);
	tensor.setValues({{{{1}, {2}}, {{3}, {4}}}});
	
	Eigen::MatrixXd result;
	EXPECT_NO_THROW(result = NEURAL_NETWORK::TensorUtils::Tensor4DToMatrix(tensor));
	
	EXPECT_EQ(result.rows(), 1); 
	EXPECT_EQ(result.cols(), 4); 
	
	EXPECT_NEAR(result(0, 0), 1, tolerance);
	EXPECT_NEAR(result(0, 1), 2, tolerance);
	EXPECT_NEAR(result(0, 2), 3, tolerance);
	EXPECT_NEAR(result(0, 3), 4, tolerance);
}

TEST_F(TensorUtilsTest, ConversionRoundTrip)
{
	Eigen::MatrixXd original(1, 4);
	original << 1.5, -2.3, 0.0, 4.7;
	
	Eigen::Tensor<double, 4> tensor(1, 2, 2, 1);
	NEURAL_NETWORK::TensorUtils::MatrixToTensor4D(original, tensor, 1, 2, 2, 1);
	
	Eigen::MatrixXd recovered = NEURAL_NETWORK::TensorUtils::Tensor4DToMatrix(tensor);
	
	EXPECT_EQ(recovered.rows(), original.rows());
	EXPECT_EQ(recovered.cols(), original.cols());
	
	for (int i = 0; i < original.rows(); ++i) {
		for (int j = 0; j < original.cols(); ++j) {
			EXPECT_NEAR(recovered(i, j), original(i, j), tolerance);
		}
	}
}

TEST_F(TensorUtilsTest, Im2ColFunctionExists)
{
	Eigen::Tensor<double, 4> tensor(1, 3, 3, 1);
	tensor.setZero();
	
	EXPECT_NO_THROW({
		Eigen::MatrixXd result = NEURAL_NETWORK::TensorUtils::im2col(
			tensor, 2, 2, 0, 0, 1, 1);
		
		EXPECT_GT(result.rows(), 0);
		EXPECT_GT(result.cols(), 0);
	});
}

TEST_F(TensorUtilsTest, Col2ImFunctionExists)
{
	Eigen::MatrixXd col_matrix = Eigen::MatrixXd::Zero(4, 4);
	Eigen::Tensor<double, 4> tensor(1, 3, 3, 1);
	
	EXPECT_NO_THROW({
		NEURAL_NETWORK::TensorUtils::col2im(
			col_matrix, tensor, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1);
	});
}