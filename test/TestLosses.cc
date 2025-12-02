#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>
#include "LossCategoricalCrossentropy.h"
#include "LossBinaryCrossEntropy.h"
#include "LossMeanSquaredError.h"
#include "LossMeanAbsoluteError.h"
#include "LayerDense.h"

class LossTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		predictions = Eigen::MatrixXd(3, 3);
		predictions << 0.7, 0.2, 0.1,
						0.1, 0.8, 0.1,
						0.2, 0.3, 0.5;

		true_labels = Eigen::MatrixXd(3, 3);
		true_labels << 1, 0, 0,
						0, 1, 0,
						0, 0, 1;

		regression_targets = Eigen::MatrixXd(3, 1);
		regression_targets << 2.5, 1.8, 3.2;

		regression_predictions = Eigen::MatrixXd(3, 1);
		regression_predictions << 2.3, 2.1, 3.0;

		binary_predictions = Eigen::MatrixXd(4, 1);
		binary_predictions << 0.9, 0.3, 0.7, 0.1;

		binary_targets = Eigen::MatrixXd(4, 1);
		binary_targets << 1, 0, 1, 0;
	}

	Eigen::MatrixXd predictions;
	Eigen::MatrixXd true_labels;
	Eigen::MatrixXd regression_targets;
	Eigen::MatrixXd regression_predictions;
	Eigen::MatrixXd binary_predictions;
	Eigen::MatrixXd binary_targets;
	const double tolerance = 1e-10;
};

class CategoricalCrossEntropyTest : public LossTest {};

TEST_F(CategoricalCrossEntropyTest, CalculateReturnsPositiveLoss) 
{
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;
	loss.CalculateLoss(predictions, true_labels);
	double loss_value = loss.GetLoss();

	EXPECT_GT(loss_value, 0.0);
	EXPECT_TRUE(std::isfinite(loss_value));
}

TEST_F(CategoricalCrossEntropyTest, PerfectPredictionsGiveLowLoss) 
{
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;

	loss.CalculateLoss(true_labels, true_labels);
	double perfect_loss = loss.GetLoss();

	EXPECT_LT(perfect_loss, 0.01);
	EXPECT_GT(perfect_loss, 0.0);
}

TEST_F(CategoricalCrossEntropyTest, WorstPredictionsGiveHighLoss) 
{
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;

	Eigen::MatrixXd worst_predictions(3, 3);
	worst_predictions << 0.05, 0.9 ,  0.05,
							0.9 , 0.05,  0.05,
							0.45, 0.5 ,  0.05;

	loss.CalculateLoss(predictions, true_labels);
	double good_loss = loss.GetLoss();

	loss.CalculateLoss(worst_predictions, true_labels);
	double bad_loss = loss.GetLoss();

	EXPECT_GT(bad_loss, good_loss);
}

TEST_F(CategoricalCrossEntropyTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;

	loss.CalculateLoss(predictions, true_labels);

	loss.backward(predictions, true_labels);
	const auto& gradients = loss.GetDInput();

	EXPECT_EQ(gradients.rows(), predictions.rows());
	EXPECT_EQ(gradients.cols(), predictions.cols());

	EXPECT_GT(gradients.cwiseAbs().sum(), 0.0);
}

TEST_F(CategoricalCrossEntropyTest, ClippingPreventsLogZero) 
{
	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;

	Eigen::MatrixXd predictions_with_zeros(2, 3);
	predictions_with_zeros << 0.0, 0.5, 0.5,
								0.3, 0.0, 0.7;

	Eigen::MatrixXd targets(2, 3);
	targets << 1, 0, 0,
				0, 1, 0;

	loss.CalculateLoss(predictions_with_zeros, targets);
	double loss_value = loss.GetLoss();

	EXPECT_TRUE(std::isfinite(loss_value));
	EXPECT_GT(loss_value, 0.0);
}

class BinaryCrossEntropyTest : public LossTest {};

TEST_F(BinaryCrossEntropyTest, CalculateReturnsPositiveLoss) 
{
	NEURAL_NETWORK::LossBinaryCrossEntropy loss;
	loss.CalculateLoss(binary_predictions, binary_targets);
	double loss_value = loss.GetLoss();

	EXPECT_GT(loss_value, 0.0);
	EXPECT_TRUE(std::isfinite(loss_value));
}

TEST_F(BinaryCrossEntropyTest, PerfectPredictionsGiveLowLoss) 
{
	NEURAL_NETWORK::LossBinaryCrossEntropy loss;

	Eigen::MatrixXd perfect_preds(4, 1);
	perfect_preds << 0.999, 0.001, 0.999, 0.001;  

	loss.CalculateLoss(perfect_preds, binary_targets);
	double perfect_loss = loss.GetLoss();

	loss.CalculateLoss(binary_predictions, binary_targets);
	double imperfect_loss = loss.GetLoss();

	EXPECT_LT(perfect_loss, imperfect_loss);
}

TEST_F(BinaryCrossEntropyTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::LossBinaryCrossEntropy loss;

	loss.CalculateLoss(binary_predictions, binary_targets);
	loss.backward(binary_predictions, binary_targets);
	const auto& gradients = loss.GetDInput();

	EXPECT_EQ(gradients.rows(), binary_predictions.rows());
	EXPECT_EQ(gradients.cols(), binary_predictions.cols());

	EXPECT_GT(gradients.cwiseAbs().sum(), 0.0);
}

TEST_F(BinaryCrossEntropyTest, GradientSignCorrectness) 
{
	NEURAL_NETWORK::LossBinaryCrossEntropy loss;

	loss.CalculateLoss(binary_predictions, binary_targets);
	loss.backward(binary_predictions, binary_targets);
	const auto& gradients = loss.GetDInput();

	for (int i = 0; i < binary_predictions.rows(); i++) 
	{
		double pred = binary_predictions(i, 0);
		double target = binary_targets(i, 0);
		double grad = gradients(i, 0);

		if (pred < target) 
		{
			EXPECT_LT(grad, 0.0) << "Gradient should be negative when prediction < target";
		} 
		else if (pred > target) 
		{
			EXPECT_GT(grad, 0.0) << "Gradient should be positive when prediction > target";
		}
	}
}

class MeanSquaredErrorTest : public LossTest {};

TEST_F(MeanSquaredErrorTest, CalculateCorrectness) 
{
	NEURAL_NETWORK::LossMeanSquaredError loss;
	loss.CalculateLoss(regression_predictions, regression_targets);
	double loss_value = loss.GetLoss();

	Eigen::MatrixXd diff = regression_predictions - regression_targets;
	double expected_mse = diff.cwiseProduct(diff).mean();

	EXPECT_NEAR(loss_value, expected_mse, tolerance);
	EXPECT_GE(loss_value, 0.0);  
}

TEST_F(MeanSquaredErrorTest, PerfectPredictionsGiveZeroLoss) 
{
	NEURAL_NETWORK::LossMeanSquaredError loss;
	loss.CalculateLoss(regression_targets, regression_targets);
	double perfect_loss = loss.GetLoss();

	EXPECT_NEAR(perfect_loss, 0.0, tolerance);
}

TEST_F(MeanSquaredErrorTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::LossMeanSquaredError loss;

	loss.CalculateLoss(regression_predictions, regression_targets);
	loss.backward(regression_predictions, regression_targets);
	const auto& gradients = loss.GetDInput();

	Eigen::MatrixXd expected_grad = 2.0 * (regression_predictions - regression_targets) / regression_predictions.rows();

	EXPECT_TRUE(gradients.isApprox(expected_grad, tolerance));
}

TEST_F(MeanSquaredErrorTest, ScaleInvariance) 
{
	NEURAL_NETWORK::LossMeanSquaredError loss;

	double scale = 10.0;
	Eigen::MatrixXd scaled_preds = scale * regression_predictions;
	Eigen::MatrixXd scaled_targets = scale * regression_targets;

	loss.CalculateLoss(regression_predictions, regression_targets);
	double original_loss = loss.GetLoss();

	loss.CalculateLoss(scaled_preds, scaled_targets);
	double scaled_loss = loss.GetLoss();

	EXPECT_NEAR(scaled_loss, scale * scale * original_loss, tolerance);
}

class MeanAbsoluteErrorTest : public LossTest {};

TEST_F(MeanAbsoluteErrorTest, CalculateCorrectness) 
{
	NEURAL_NETWORK::LossMeanAbsoluteError loss;
	loss.CalculateLoss(regression_predictions, regression_targets);
	double loss_value = loss.GetLoss();

	Eigen::MatrixXd diff = regression_predictions - regression_targets;
	double expected_mae = diff.cwiseAbs().mean();

	EXPECT_NEAR(loss_value, expected_mae, tolerance);
	EXPECT_GE(loss_value, 0.0);  
}

TEST_F(MeanAbsoluteErrorTest, PerfectPredictionsGiveZeroLoss) 
{
	NEURAL_NETWORK::LossMeanAbsoluteError loss;
	loss.CalculateLoss(regression_targets, regression_targets);
	double perfect_loss = loss.GetLoss();

	EXPECT_NEAR(perfect_loss, 0.0, tolerance);
}

TEST_F(MeanAbsoluteErrorTest, BackwardPassCorrectness) 
{
	NEURAL_NETWORK::LossMeanAbsoluteError loss;

	loss.CalculateLoss(regression_predictions, regression_targets);
	loss.backward(regression_predictions, regression_targets);
	const auto& gradients = loss.GetDInput();

	EXPECT_EQ(gradients.rows(), regression_predictions.rows());
	EXPECT_EQ(gradients.cols(), regression_predictions.cols());

	EXPECT_GT(gradients.cwiseAbs().sum(), 0.0);

	for (int i = 0; i < regression_predictions.rows(); i++) 
	{
		for (int j = 0; j < regression_predictions.cols(); j++) 
		{
			double diff = regression_predictions(i, j) - regression_targets(i, j);
			double grad = gradients(i, j);

			EXPECT_TRUE(std::abs(grad) > 0.0);
		}
	}
}

TEST_F(MeanAbsoluteErrorTest, OutlierRobustness) 
{
	NEURAL_NETWORK::LossMeanAbsoluteError mae_loss;
	NEURAL_NETWORK::LossMeanSquaredError mse_loss;

	Eigen::MatrixXd normal_preds(3, 1);
	normal_preds << 1.0, 2.0, 3.0;

	Eigen::MatrixXd normal_targets(3, 1);
	normal_targets << 1.1, 2.1, 3.1;

	Eigen::MatrixXd outlier_preds(3, 1);
	outlier_preds << 1.0, 2.0, 10.0;  

	mae_loss.CalculateLoss(normal_preds, normal_targets);
	double normal_mae = mae_loss.GetLoss();

	mse_loss.CalculateLoss(normal_preds, normal_targets);
	double normal_mse = mse_loss.GetLoss();

	mae_loss.CalculateLoss(outlier_preds, normal_targets);
	double outlier_mae = mae_loss.GetLoss();

	mse_loss.CalculateLoss(outlier_preds, normal_targets);
	double outlier_mse = mse_loss.GetLoss();

	double mae_ratio = outlier_mae / normal_mae;
	double mse_ratio = outlier_mse / normal_mse;

	EXPECT_LT(mae_ratio, mse_ratio);
}

TEST_F(CategoricalCrossEntropyTest, RegularizationLossComputed)
{
	auto layer = std::make_shared<NEURAL_NETWORK::LayerDense>(3, 3,
		0.05, 0.1, 0.02, 0.03);
	Eigen::MatrixXd weights(3, 3);
	weights <<  0.2, -0.1,  0.05,
		        0.3,  0.4, -0.2,
		       -0.15, 0.25, 0.1;
	Eigen::RowVectorXd biases(3);
	biases << 0.05, -0.02, 0.01;
	layer->SetParameters(weights, biases);

	NEURAL_NETWORK::LossCategoricalCrossEntropy loss;
	std::vector<std::weak_ptr<NEURAL_NETWORK::LayerBase>> layers = {layer};
	loss.RememberTrainableLayers(layers);

	loss.CalculateLoss(predictions, true_labels);
	loss.RegularizationLoss();

	double expected = 0.0;
	expected += layer->GetWeightRegularizerL1() * weights.array().abs().sum();
	expected += layer->GetWeightRegularizerL2() * weights.array().square().sum();
	expected += layer->GetBiasRegularizerL1() * biases.array().abs().sum();
	expected += layer->GetBiasRegularizerL2() * biases.array().square().sum();

	double reg_loss = loss.GetRegularizationLoss();
	EXPECT_NEAR(reg_loss, expected, 1e-12);
}
