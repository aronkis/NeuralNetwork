#include "Helpers.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 

void NEURAL_NETWORK::Helpers::ReadSpiralIntoEigen(const std::string& filename,
												  Eigen::MatrixXd& coordinates,
												  Eigen::MatrixXi& classes) 
{
	std::ifstream input_file(filename);

	if (!input_file.is_open()) 
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		coordinates.resize(0, 0); 
		classes.resize(0, 0);
		return;
	}

	std::vector<double> x_coords;
	std::vector<double> y_coords;
	std::vector<int> class_values;

	std::string line;
	int line_num = 0;

	while (std::getline(input_file, line)) 
	{
		line_num++;
		std::stringstream ss(line);
		double x_val, y_val;
		int class_val;

		if (ss >> x_val >> y_val >> class_val) 
		{
			x_coords.push_back(x_val);
			y_coords.push_back(y_val);
			class_values.push_back(class_val);
		} 
		else 
		{
			std::cerr << "Warning: Skipping malformed line " << line_num << ": '" << line
					  << "' (expected 2 doubles and 1 integer)" << std::endl;
		}
	}

	input_file.close();
	
	long num_rows = x_coords.size();

	if (num_rows > 0) {
		coordinates.resize(num_rows, 2); 
		classes.resize(num_rows, 1);        
		
		for (long i = 0; i < num_rows; i++) 
		{
			coordinates(i, 0) = x_coords[i]; 
			coordinates(i, 1) = y_coords[i]; 
			classes(i) = class_values[i];    
		}
	} 
	else 
	{
		std::cout << "No valid data lines found in the file." << std::endl;
		coordinates.resize(0, 0);
		classes.resize(0, 0);
	}
}

void NEURAL_NETWORK::Helpers::Read1DIntoEigen(const std::string& filename,
												  Eigen::MatrixXd& input,
												  Eigen::MatrixXd& output) 
{
	std::ifstream input_file(filename);

	if (!input_file.is_open()) 
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		input.resize(0, 0); 
		output.resize(0, 0);
		return;
	}

	std::vector<double> input_values;
	std::vector<double> output_values;

	std::string line;

	while (std::getline(input_file, line)) 
	{
		std::stringstream ss(line);
		double in_val, out_val;

		if (ss >> in_val >> out_val) 
		{
			input_values.push_back(in_val);
			output_values.push_back(out_val);
		} 
		else 
		{
			std::cerr << "Warning: Skipping malformed line: '" << line
					  << "' (expected 2 doubles)" << std::endl;
		}
	}

	input_file.close();
	
	long num_rows = input_values.size();

	if (num_rows > 0) {
		input.resize(num_rows, 1);  
		output.resize(num_rows, 1); 
		
		for (long i = 0; i < num_rows; i++) 
		{
			input(i, 0) = input_values[i];  
			output(i, 0) = output_values[i]; 
		}
	} 
	else 
	{
		std::cout << "No valid data lines found in the file." << std::endl;
		input.resize(0, 0);
		output.resize(0, 0);
	}
}

double NEURAL_NETWORK::Helpers::CalculateAccuracy(const Eigen::MatrixXd& output, 
												  Eigen::MatrixXi& targets) 
{
	Eigen::VectorXi predictions(output.rows());
	
	for (int i = 0; i < output.rows(); i++) 
	{
		output.row(i).maxCoeff(&predictions(i));
	}
	
	if (targets.cols() == 2)
	{
		targets = targets.colwise().maxCoeff();
	}

	double accuracy = (predictions.array() == targets.array()).cast<double>().mean();
	return accuracy;
}

double NEURAL_NETWORK::Helpers::CalculateRegressionAccuracy(const Eigen::MatrixXd& output, 
															const Eigen::MatrixXd& targets, 
															double epsilon) 
{
	Eigen::ArrayXXd within = ((output.array() - targets.array()).abs() < epsilon).cast<double>();
	return within.mean();
}

double NEURAL_NETWORK::Helpers::CalculateEpsilon(const Eigen::MatrixXd& target) 
{
    double mean = target.mean();
    double variance = (target.array() - mean).square().mean();
    double std_dev = std::sqrt(variance);
    return std_dev / 250.0;
}

Eigen::MatrixXd NEURAL_NETWORK::Helpers::MatrixSquare(const Eigen::MatrixXd& matrix) 
{
	return matrix.array().square().matrix();
}

Eigen::ArrayXXd NEURAL_NETWORK::Helpers::MatrixSquareRootToArray(const Eigen::MatrixXd& matrix) 
{
	return matrix.array().sqrt();
}