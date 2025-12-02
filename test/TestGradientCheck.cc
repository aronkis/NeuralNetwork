#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "Convolution.h"
#include "MaxPooling.h"
#include "LayerDense.h"
#include "ActivationReLU.h"

class GradientCheckTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		epsilon = 1e-5;
		tolerance = 1e-3; 
		test_input_2x2 = Eigen::MatrixXd(1, 4);
		test_input_2x2 << 0.1, 0.2, 0.3, 0.4;
		
		test_input_3x3 = Eigen::MatrixXd(1, 9);
		test_input_3x3 << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
		test_batch_2x2 = Eigen::MatrixXd(2, 4);
		test_batch_2x2 << 0.1, 0.2, 0.3, 0.4,
						  0.5, 0.6, 0.7, 0.8;
	}
	
	Eigen::MatrixXd test_input_2x2;
	Eigen::MatrixXd test_input_3x3;
	Eigen::MatrixXd test_batch_2x2;
	double epsilon;
	double tolerance;
};

TEST_F(GradientCheckTest, ConvolutionWeightGradients)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1);
	
	conv.forward(test_input_2x2, true);
	
	const auto& output = conv.GetOutput();
	double original_loss = 0.5 * output.array().square().sum();
	
	Eigen::MatrixXd d_output = output; 
	
	conv.backward(d_output);
	const auto& analytical_d_weights = conv.GetDWeightsTensor();
	
	auto weights = conv.GetWeightsTensor();
	
	for (int i = 0; i < weights.dimension(0); ++i) {
		for (int j = 0; j < weights.dimension(1); ++j) {
			for (int k = 0; k < weights.dimension(2); ++k) {
				for (int l = 0; l < weights.dimension(3); ++l) {
					double original_weight = weights(i, j, k, l);
					auto loss_func = [&](double w) -> double {
						
						auto temp_weights = weights;
						temp_weights(i, j, k, l) = w;
						conv.SetWeightsTensor(temp_weights);
						
						conv.forward(test_input_2x2, false);
						const auto& temp_output = conv.GetOutput();
						
						return 0.5 * temp_output.array().square().sum();
					};
					
					double numerical_grad = NumericalGradient(loss_func, original_weight);
					double analytical_grad = analytical_d_weights(i, j, k, l);
					
					EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
						<< "Weight gradient mismatch at (" << i << "," << j << "," << k << "," << l << ")"
						<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
					
					weights(i, j, k, l) = original_weight;
					conv.SetWeightsTensor(weights);
				}
			}
		}
	}
}

TEST_F(GradientCheckTest, ConvolutionBiasGradients)
{
	NEURAL_NETWORK::Convolution conv(2, 2, 2, 2, 2, 1, false, 1, 1);
	
	conv.forward(test_input_2x2, true);
	
	const auto& output = conv.GetOutput();
	Eigen::MatrixXd d_output = output; 
	
	conv.backward(d_output);
	const auto& analytical_d_biases = conv.GetDBiasesVector();
	
	auto biases = conv.GetBiasesVector();
	
	for (int i = 0; i < biases.size(); ++i) {
		double original_bias = biases(i);
		
		auto loss_func = [&](double b) -> double {
			
			auto temp_biases = biases;
			temp_biases(i) = b;
			conv.SetBiasesVector(temp_biases);
			
			conv.forward(test_input_2x2, false);
			const auto& temp_output = conv.GetOutput();
			
			return 0.5 * temp_output.array().square().sum();
		};
		
		double numerical_grad = NumericalGradient(loss_func, original_bias);
		double analytical_grad = analytical_d_biases(i);
		
		EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
			<< "Bias gradient mismatch at index " << i
			<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
		
		biases(i) = original_bias;
		conv.SetBiasesVector(biases);
	}
}

TEST_F(GradientCheckTest, ConvolutionInputGradients)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 3, 3, 1, false, 1, 1);
	
	conv.forward(test_input_3x3, true);
	
	const auto& output = conv.GetOutput();
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	
	conv.backward(d_output);
	const auto& analytical_d_input = conv.GetDInput();
	
	auto input = test_input_3x3;
	
	for (int i = 0; i < input.rows(); ++i) {
		for (int j = 0; j < input.cols(); ++j) {
			double original_input = input(i, j);
			
			auto loss_func = [&](double x) -> double {
				auto temp_input = input;
				temp_input(i, j) = x;
				
				conv.forward(temp_input, false);
				const auto& temp_output = conv.GetOutput();
				
				return temp_output.sum();
			};
			
			double numerical_grad = NumericalGradient(loss_func, original_input);
			double analytical_grad = analytical_d_input(i, j);
			
			EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
				<< "Input gradient mismatch at (" << i << "," << j << ")"
				<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
		}
	}
}

TEST_F(GradientCheckTest, MaxPoolingInputGradients)
{
	NEURAL_NETWORK::MaxPooling pool(1, 2, 2, 2, 1, 1);
	
	pool.forward(test_input_2x2, true);
	
	const auto& output = pool.GetOutput();
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	
	pool.backward(d_output);
	const auto& analytical_d_input = pool.GetDInput();
	
	auto input = test_input_2x2;
	
	for (int i = 0; i < input.rows(); ++i) {
		for (int j = 0; j < input.cols(); ++j) {
			double original_input = input(i, j);
			
			auto loss_func = [&](double x) -> double {
				auto temp_input = input;
				temp_input(i, j) = x;
				
				pool.forward(temp_input, false);
				const auto& temp_output = pool.GetOutput();
				
				return temp_output.sum();
			};
			
			double numerical_grad = NumericalGradient(loss_func, original_input);
			double analytical_grad = analytical_d_input(i, j);
			
			double effective_tolerance = (std::abs(analytical_grad) < 1e-8) ? 1e-3 : tolerance;
			
			EXPECT_NEAR(numerical_grad, analytical_grad, effective_tolerance)
				<< "MaxPooling input gradient mismatch at (" << i << "," << j << ")"
				<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
		}
	}
}

TEST_F(GradientCheckTest, LayerDenseWeightGradients)
{
	NEURAL_NETWORK::LayerDense dense(4, 2);
	
	dense.forward(test_input_2x2, true);
	
	const auto& output = dense.GetOutput();
	Eigen::MatrixXd d_output = output; 
	
	dense.backward(d_output);
	const auto& analytical_d_weights = dense.GetDWeights();
	
	auto weights = dense.GetWeights();
	
	for (int i = 0; i < weights.rows(); ++i) {
		for (int j = 0; j < weights.cols(); ++j) {
			double original_weight = weights(i, j);
			
			auto loss_func = [&](double w) -> double {
				auto temp_weights = weights;
				temp_weights(i, j) = w;
				dense.SetWeights(temp_weights);
				
				dense.forward(test_input_2x2, false);
				const auto& temp_output = dense.GetOutput();
				
				return 0.5 * temp_output.array().square().sum();
			};
			
			double numerical_grad = NumericalGradient(loss_func, original_weight);
			double analytical_grad = analytical_d_weights(i, j);
			
			EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
				<< "Dense weight gradient mismatch at (" << i << "," << j << ")"
				<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
			
			weights(i, j) = original_weight;
			dense.SetWeights(weights);
		}
	}
}

TEST_F(GradientCheckTest, ConvolutionWithRegularization)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1, 0.1, 0.1); 
	
	auto weights = conv.GetWeightsTensor();
	for (int i = 0; i < weights.dimension(0); ++i) {
		for (int j = 0; j < weights.dimension(1); ++j) {
			for (int k = 0; k < weights.dimension(2); ++k) {
				for (int l = 0; l < weights.dimension(3); ++l) {
					weights(i, j, k, l) = 0.5 * (i + j + k + l + 1);
				}
			}
		}
	}
	conv.SetWeightsTensor(weights);
	
	conv.forward(test_input_2x2, true);
	
	const auto& output = conv.GetOutput();
	Eigen::MatrixXd d_output = output; 
	
	conv.backward(d_output);
	const auto& analytical_d_weights = conv.GetDWeightsTensor();
	
	for (int i = 0; i < weights.dimension(0); ++i) {
		for (int j = 0; j < weights.dimension(1); ++j) {
			for (int k = 0; k < weights.dimension(2); ++k) {
				for (int l = 0; l < weights.dimension(3); ++l) {
					double original_weight = weights(i, j, k, l);
					
					auto loss_func = [&](double w) -> double {
						auto temp_weights = weights;
						temp_weights(i, j, k, l) = w;
						conv.SetWeightsTensor(temp_weights);
						
						conv.forward(test_input_2x2, false);
						const auto& temp_output = conv.GetOutput();
						
						double data_loss = 0.5 * temp_output.array().square().sum();
						
						double reg_loss = conv.CalculateRegularizationLoss();
						
						return data_loss + reg_loss;
					};
					
					double numerical_grad = NumericalGradient(loss_func, original_weight);
					double analytical_grad = analytical_d_weights(i, j, k, l);
					
					EXPECT_NEAR(numerical_grad, analytical_grad, tolerance * 10)
						<< "Regularized weight gradient mismatch at (" << i << "," << j << "," << k << "," << l << ")"
						<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
					
					weights(i, j, k, l) = original_weight;
					conv.SetWeightsTensor(weights);
				}
			}
		}
	}
}

TEST_F(GradientCheckTest, ConvolutionBatchGradients)
{
	NEURAL_NETWORK::Convolution conv(1, 2, 2, 2, 2, 1, false, 1, 1);
	
	conv.forward(test_batch_2x2, true);
	
	const auto& output = conv.GetOutput();
	Eigen::MatrixXd d_output = Eigen::MatrixXd::Ones(output.rows(), output.cols());
	
	conv.backward(d_output);
	const auto& analytical_d_input = conv.GetDInput();
	
	auto input = test_batch_2x2;
	
	std::vector<std::pair<int, int>> test_positions = {{0, 0}, {0, 3}, {1, 1}, {1, 2}};
	
	for (const auto& pos : test_positions) {
		int i = pos.first;
		int j = pos.second;
		double original_input = input(i, j);
		
		auto loss_func = [&](double x) -> double {
			auto temp_input = input;
			temp_input(i, j) = x;
			
			conv.forward(temp_input, false);
			const auto& temp_output = conv.GetOutput();
			
			return temp_output.sum();
		};
		
		double numerical_grad = NumericalGradient(loss_func, original_input);
		double analytical_grad = analytical_d_input(i, j);
		
		EXPECT_NEAR(numerical_grad, analytical_grad, tolerance)
			<< "Batch input gradient mismatch at (" << i << "," << j << ")"
			<< " Numerical: " << numerical_grad << " Analytical: " << analytical_grad;
	}
}