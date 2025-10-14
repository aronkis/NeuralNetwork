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
    // Test basic matrix to tensor conversion
    Eigen::MatrixXd matrix(2, 8); // 2 samples, 2x2x2 data
    matrix << 1, 2, 3, 4, 5, 6, 7, 8,
              9, 10, 11, 12, 13, 14, 15, 16;
    
    Eigen::Tensor<double, 4> tensor(2, 2, 2, 2);
    
    EXPECT_NO_THROW(NEURAL_NETWORK::TensorUtils::MatrixToTensor4D(
        matrix, tensor, 2, 2, 2, 2));
    
    // Check tensor dimensions
    EXPECT_EQ(tensor.dimension(0), 2); // batch
    EXPECT_EQ(tensor.dimension(1), 2); // height
    EXPECT_EQ(tensor.dimension(2), 2); // width
    EXPECT_EQ(tensor.dimension(3), 2); // channels
}

TEST_F(TensorUtilsTest, Tensor4DToMatrixBasic)
{
    // Test basic tensor to matrix conversion
    Eigen::Tensor<double, 4> tensor(1, 2, 2, 1);
    tensor.setValues({{{{1}, {2}}, {{3}, {4}}}});
    
    Eigen::MatrixXd result;
    EXPECT_NO_THROW(result = NEURAL_NETWORK::TensorUtils::Tensor4DToMatrix(tensor));
    
    // Check matrix dimensions
    EXPECT_EQ(result.rows(), 1); // batch size
    EXPECT_EQ(result.cols(), 4); // 2*2*1 = 4
    
    // Check values
    EXPECT_NEAR(result(0, 0), 1, tolerance);
    EXPECT_NEAR(result(0, 1), 2, tolerance);
    EXPECT_NEAR(result(0, 2), 3, tolerance);
    EXPECT_NEAR(result(0, 3), 4, tolerance);
}

TEST_F(TensorUtilsTest, ConversionRoundTrip)
{
    // Test that matrix->tensor->matrix preserves data
    Eigen::MatrixXd original(1, 4);
    original << 1.5, -2.3, 0.0, 4.7;
    
    Eigen::Tensor<double, 4> tensor(1, 2, 2, 1);
    NEURAL_NETWORK::TensorUtils::MatrixToTensor4D(original, tensor, 1, 2, 2, 1);
    
    Eigen::MatrixXd recovered = NEURAL_NETWORK::TensorUtils::Tensor4DToMatrix(tensor);
    
    // Should be identical
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
    // Just test that im2col function exists and can be called
    Eigen::Tensor<double, 4> tensor(1, 3, 3, 1);
    tensor.setZero();
    
    EXPECT_NO_THROW({
        Eigen::MatrixXd result = NEURAL_NETWORK::TensorUtils::im2col(
            tensor, 2, 2, 0, 0, 1, 1);
        // Just check it returns something reasonable
        EXPECT_GT(result.rows(), 0);
        EXPECT_GT(result.cols(), 0);
    });
}

TEST_F(TensorUtilsTest, Col2ImFunctionExists)
{
    // Just test that col2im function exists and can be called
    Eigen::MatrixXd col_matrix = Eigen::MatrixXd::Zero(4, 4);
    Eigen::Tensor<double, 4> tensor(1, 3, 3, 1);
    
    EXPECT_NO_THROW({
        NEURAL_NETWORK::TensorUtils::col2im(
            col_matrix, tensor, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1);
    });
}