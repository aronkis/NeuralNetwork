#include "Helpers.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 

void Helpers::ReadSpiralIntoEigen(const std::string& filename,
                                  Eigen::MatrixXd& coordinates,
                                  Eigen::RowVectorXi& classes) 
{
    std::ifstream inputFile(filename);

    if (!inputFile.is_open()) 
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        coordinates.resize(0, 0); 
        classes.resize(0);
        return;
    }

    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<int> class_values;

    std::string line;
    int lineNum = 0;

    while (std::getline(inputFile, line)) 
    {
        lineNum++;
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
            std::cerr << "Warning: Skipping malformed line " << lineNum << ": '" << line
                      << "' (expected 2 doubles and 1 integer)" << std::endl;
        }
    }
    inputFile.close();
    
    long num_rows = x_coords.size();

    if (num_rows > 0) {
        coordinates.resize(num_rows, 2); 
        classes.resize(num_rows);        
        
        for (long i = 0; i < num_rows; ++i) 
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
        classes.resize(0);
    }
}