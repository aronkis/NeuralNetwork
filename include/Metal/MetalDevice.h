#ifndef __METAL_DEVICE_H__
#define __METAL_DEVICE_H__

#ifdef METAL_ENABLED

// Forward declarations for Objective-C types
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#else
typedef struct objc_object* id;
#endif

#include <Eigen/Dense>

namespace NEURAL_NETWORK {
namespace Metal {

class MetalDevice {
public:
    static MetalDevice& getInstance();
    
    bool isMetalAvailable() const;
    
    // Matrix operations
    void matrixMultiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);
    void matrixAdd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);
    void matrixSubtract(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);
    void matrixElementwiseMultiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);
    void matrixElementwiseDivide(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);
    
    // Activation functions
    void relu(const Eigen::MatrixXd& input, Eigen::MatrixXd& output);
    void reluBackward(const Eigen::MatrixXd& gradOutput, const Eigen::MatrixXd& input, Eigen::MatrixXd& gradInput);
    void softmax(const Eigen::MatrixXd& input, Eigen::MatrixXd& output);
    
    // Vector operations
    void vectorAdd(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C);
    void vectorSubtract(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C);
    void vectorElementwiseMultiply(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C);
    void vectorElementwiseDivide(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C);
    
private:
    MetalDevice();
    ~MetalDevice();
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    
    void initializeMetal();
    id createBufferFromMatrix(const Eigen::MatrixXd& matrix);
    id createBufferFromVector(const Eigen::RowVectorXd& vector);
    void copyMatrixFromBuffer(id buffer, Eigen::MatrixXd& matrix);
    void copyVectorFromBuffer(id buffer, Eigen::RowVectorXd& vector);
    
    id device_;
    id commandQueue_;
    id matrixMultiplicationKernel_;
    bool metalAvailable_;
};

} // namespace Metal
} // namespace NEURAL_NETWORK

#else
// Empty class when Metal is not available
namespace NEURAL_NETWORK {
namespace Metal {
class MetalDevice {
public:
    static MetalDevice& getInstance() {
        static MetalDevice instance;
        return instance;
    }
    bool isMetalAvailable() const { return false; }
};
} // namespace Metal
} // namespace NEURAL_NETWORK

#endif // METAL_ENABLED

#endif // __METAL_DEVICE_H__
