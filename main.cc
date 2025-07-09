#include <iostream>
#include <cmath>

#include <matplot/matplot.h>
#include <Eigen/Dense>

#include "LayerDense.h"
#include "Helpers.h"
 
int main() {

  std::string filename = "../points.txt"; 

    Eigen::MatrixXd X;      
    Eigen::RowVectorXi y;      

    Helpers::ReadSpiralIntoEigen(filename, X, y);

    if (X.rows() > 0) 
    {
        std::cout << "Successfully read " << X.rows() << " points." << std::endl;
    } 
    else 
    {
        std::cout << "Could not read any data or file not found." << std::endl;
    }


    LayerDense l1(2, 3);

    l1.forward(X);

    std::cout << l1.GetOutput().topRows(5) << std::endl;

    return 0;
}