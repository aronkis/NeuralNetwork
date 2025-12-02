#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "AccuracyCategorical.h"
#include "AccuracyRegression.h"
 
class AccuracyTest : public ::testing::Test 
{
protected:
	void SetUp() override 
	{
		classification_predictions = Eigen::MatrixXd(5, 3);
		classification_predictions << 0.8, 0.1, 0.1,
										0.2, 0.7, 0.1,
										0.1, 0.2, 0.7,
										0.3, 0.3, 0.4,
										0.6, 0.3, 0.1;

		classification_targets = Eigen::MatrixXd(5, 3);
		classification_targets << 1, 0, 0,
									0, 1, 0,
									0, 0, 1,
									0, 0, 1,
									0, 1, 0;

		regression_predictions = Eigen::MatrixXd(6, 1);
		regression_predictions << 2.1,
									5.2,
									8.8,
									1.5,
									12.0, 
									7.3;

		regression_targets = Eigen::MatrixXd(6, 1);
		regression_targets << 2.0,
								5.0,
								9.0,
								10.0,
								12.0,
								7.0;
	}

	Eigen::MatrixXd classification_predictions;
	Eigen::MatrixXd classification_targets;
	Eigen::MatrixXd regression_predictions;
	Eigen::MatrixXd regression_targets;
	const double tolerance = 1e-10;
};

class CategoricalAccuracyTest : public AccuracyTest {};

TEST_F(CategoricalAccuracyTest, PerfectClassificationGivesFullAccuracy) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;
	
	Eigen::MatrixXd perfect_predictions = classification_targets;

	accuracy.Calculate(perfect_predictions, classification_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST_F(CategoricalAccuracyTest, CompletelyWrongClassificationGivesZeroAccuracy) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd wrong_predictions(5, 3);
	wrong_predictions << 0, 1, 0,
							1, 0, 0,
							1, 0, 0,
							1, 0, 0,
							1, 0, 0;

	accuracy.Calculate(wrong_predictions, classification_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST_F(CategoricalAccuracyTest, PartialCorrectClassification) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	accuracy.Calculate(classification_predictions, classification_targets);
	double result = accuracy.GetAccuracy();
	double expected_accuracy = 4.0 / 5.0;
	EXPECT_DOUBLE_EQ(result, expected_accuracy);
}

TEST_F(CategoricalAccuracyTest, HandlesOneHotTargets) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd one_hot_targets(3, 4);
	one_hot_targets << 1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;

	Eigen::MatrixXd predictions(3, 4);
	predictions << 0.9, 0.05, 0.03, 0.02,
					0.1, 0.8 , 0.05, 0.05,
					0.2, 0.1 , 0.1 , 0.6 ; 

	accuracy.Calculate(predictions, one_hot_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST_F(CategoricalAccuracyTest, HandlesClassIndexTargets) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd class_index_targets(3, 1);
	class_index_targets << 0, 1, 2;

	Eigen::MatrixXd predictions(3, 3);
	predictions << 0.8, 0.1, 0.1,
					0.1, 0.7, 0.2,
					0.1, 0.2, 0.7;

	accuracy.Calculate(predictions, class_index_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST_F(CategoricalAccuracyTest, HandlesClassImbalanceWithIndices)
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd imbalanced_predictions(6, 3);
	imbalanced_predictions << 0.92, 0.04, 0.04,
							   0.88, 0.07, 0.05,
							   0.10, 0.80, 0.10,
							   0.20, 0.60, 0.20,
							   0.95, 0.03, 0.02,
							   0.12, 0.18, 0.70;

	Eigen::MatrixXd imbalanced_targets(6, 1);
	imbalanced_targets << 0,
						  0,
						  1,
						  0,
						  0,
						  2;

	accuracy.Calculate(imbalanced_predictions, imbalanced_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 5.0 / 6.0);
}

TEST_F(CategoricalAccuracyTest, HandlesIntegerPredictionsForMulticlass)
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd integer_predictions(4, 1);
	integer_predictions << 3,
						  0,
						  1,
						  2;

	Eigen::MatrixXd one_hot_targets(4, 4);
	one_hot_targets << 0, 0, 1, 0,
					  1, 0, 0, 0,
					  0, 1, 0, 0,
					  0, 0, 1, 0;

	accuracy.Calculate(integer_predictions, one_hot_targets);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 0.75);
}

TEST_F(CategoricalAccuracyTest, SingleSampleTest) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	Eigen::MatrixXd single_prediction(1, 3);
	single_prediction << 0.1, 0.8, 0.1;

	Eigen::MatrixXd single_target(1, 3);
	single_target << 0, 1, 0;

	accuracy.Calculate(single_prediction, single_target);
	double result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(result, 1.0);

	Eigen::MatrixXd wrong_target(1, 3);
	wrong_target << 1, 0, 0;

	accuracy.Calculate(single_prediction, wrong_target);
	double wrong_result = accuracy.GetAccuracy();
	EXPECT_DOUBLE_EQ(wrong_result, 0.0);
}

class RegressionAccuracyTest : public AccuracyTest {};

TEST_F(RegressionAccuracyTest, DefaultPrecisionTest) 
{
	NEURAL_NETWORK::AccuracyRegression accuracy;

	accuracy.init(regression_targets);

	accuracy.Calculate(regression_predictions, regression_targets);
	double result = accuracy.GetAccuracy();

	double epsilon = accuracy.GetEpsilon();
	int correct_count = 0;
	for (int i = 0; i < regression_predictions.rows(); i++) 
	{
		double diff = std::abs(regression_predictions(i, 0) - regression_targets(i, 0));
		if (diff < epsilon) 
		{
			correct_count++;
		}
	}

	double expected_accuracy = static_cast<double>(correct_count) / regression_predictions.rows();
	EXPECT_DOUBLE_EQ(result, expected_accuracy);
}

TEST_F(RegressionAccuracyTest, PerfectPredictions) 
{
	NEURAL_NETWORK::AccuracyRegression accuracy;

	accuracy.init(regression_targets);
	accuracy.Calculate(regression_targets, regression_targets);
	double result = accuracy.GetAccuracy();

	EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST_F(RegressionAccuracyTest, SingleValueTest) 
{
	NEURAL_NETWORK::AccuracyRegression accuracy;

	Eigen::MatrixXd single_pred(1, 1);
	single_pred << 5.2;

	Eigen::MatrixXd single_target(1, 1);
	single_target << 5.0;

	accuracy.init(single_target);
	accuracy.Calculate(single_pred, single_target);
	double result = accuracy.GetAccuracy();

	EXPECT_GE(result, 0.0);
	EXPECT_LE(result, 1.0);
}

TEST_F(RegressionAccuracyTest, MultipleOutputRegression) 
{
	NEURAL_NETWORK::AccuracyRegression accuracy;

	Eigen::MatrixXd multi_pred(3, 2);
	multi_pred << 1.1, 2.1,
				  5.2, 3.8,
				  8.6, 7.4;

	Eigen::MatrixXd multi_target(3, 2);
	multi_target << 1.0, 2.0,
					5.0, 4.0,
					9.0, 7.8;

	accuracy.init(multi_target);
	accuracy.Calculate(multi_pred, multi_target);
	double result = accuracy.GetAccuracy();

	EXPECT_GE(result, 0.0);
	EXPECT_LE(result, 1.0);
}

TEST_F(RegressionAccuracyTest, ReinitializationUpdatesThreshold)
{
	NEURAL_NETWORK::AccuracyRegression accuracy;
	accuracy.init(regression_targets);
	double first_eps = accuracy.GetEpsilon();

	Eigen::MatrixXd wider_targets = regression_targets * 10.0;
	accuracy.init(wider_targets, true);
	double second_eps = accuracy.GetEpsilon();

	EXPECT_GT(second_eps, first_eps);
}

class BaseAccuracyTest : public AccuracyTest {};

TEST_F(BaseAccuracyTest, AccumulatedAccuracyCalculation) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	accuracy.NewPass();

	accuracy.Calculate(classification_predictions, classification_targets);
	double batch1_acc = accuracy.GetAccuracy();

	accuracy.Calculate(classification_predictions, classification_targets);
	double batch2_acc = accuracy.GetAccuracy();

	accuracy.CalculateAccumulated();
	double accumulated = accuracy.GetAccumulatedAccuracy();

	double expected = (batch1_acc + batch2_acc) / 2.0;
	EXPECT_NEAR(accumulated, expected, tolerance);
}

TEST_F(BaseAccuracyTest, NewPassResetsAccumulation) 
{
	NEURAL_NETWORK::AccuracyCategorical accuracy;

	accuracy.Calculate(classification_predictions, classification_targets);
	accuracy.CalculateAccumulated();
	double first_accumulated = accuracy.GetAccumulatedAccuracy();

	accuracy.NewPass();

	accuracy.Calculate(classification_predictions, classification_targets);
	accuracy.CalculateAccumulated();
	double second_accumulated = accuracy.GetAccumulatedAccuracy();

	EXPECT_NEAR(second_accumulated, 4.0/5.0, tolerance);
}