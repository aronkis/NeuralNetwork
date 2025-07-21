#ifdef METAL_ENABLED

#import "MetalDevice.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace NEURAL_NETWORK {
namespace Metal {

MetalDevice& MetalDevice::getInstance() {
    static MetalDevice instance;
    return instance;
}

MetalDevice::MetalDevice() : metalAvailable_(false) {
    initializeMetal();
}

MetalDevice::~MetalDevice() {
    if (device_) {
        [device_ release];
    }
    if (commandQueue_) {
        [commandQueue_ release];
    }
    if (matrixMultiplicationKernel_) {
        [matrixMultiplicationKernel_ release];
    }
}

void MetalDevice::initializeMetal() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        NSLog(@"Metal is not supported on this device");
        return;
    }
    
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
        NSLog(@"Failed to create Metal command queue");
        return;
    }
    
    metalAvailable_ = true;
    NSLog(@"Metal device initialized successfully");
}

bool MetalDevice::isMetalAvailable() const {
    return metalAvailable_;
}

id MetalDevice::createBufferFromMatrix(const Eigen::MatrixXd& matrix) {
    if (!metalAvailable_) return nil;
    
    size_t size = matrix.rows() * matrix.cols() * sizeof(float);
    
    // Convert double to float for Metal
    std::vector<float> floatData(matrix.rows() * matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            floatData[i * matrix.cols() + j] = static_cast<float>(matrix(i, j));
        }
    }
    
    return [device_ newBufferWithBytes:floatData.data() 
                                length:size 
                               options:MTLResourceStorageModeShared];
}

id MetalDevice::createBufferFromVector(const Eigen::RowVectorXd& vector) {
    if (!metalAvailable_) return nil;
    
    size_t size = vector.size() * sizeof(float);
    
    // Convert double to float for Metal
    std::vector<float> floatData(vector.size());
    for (int i = 0; i < vector.size(); ++i) {
        floatData[i] = static_cast<float>(vector(i));
    }
    
    return [device_ newBufferWithBytes:floatData.data() 
                                length:size 
                               options:MTLResourceStorageModeShared];
}

void MetalDevice::copyMatrixFromBuffer(id buffer, Eigen::MatrixXd& matrix) {
    if (!metalAvailable_ || !buffer) return;
    
    float* data = static_cast<float*>([buffer contents]);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix(i, j) = static_cast<double>(data[i * matrix.cols() + j]);
        }
    }
}

void MetalDevice::copyVectorFromBuffer(id buffer, Eigen::RowVectorXd& vector) {
    if (!metalAvailable_ || !buffer) return;
    
    float* data = static_cast<float*>([buffer contents]);
    for (int i = 0; i < vector.size(); ++i) {
        vector(i) = static_cast<double>(data[i]);
    }
}

void MetalDevice::matrixMultiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    if (!metalAvailable_) {
        // Fallback to CPU
        C = A * B;
        return;
    }
    
    // Only use GPU for larger matrices (threshold: 1M operations)
    size_t operations = A.rows() * A.cols() * B.cols();
    if (operations < 1000000) {
        C = A * B;  // Use CPU for small matrices
        return;
    }
    
    // Create MPS matrix descriptors
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:A.rows() 
                                                                       columns:A.cols() 
                                                                      rowBytes:A.cols() * sizeof(float) 
                                                                      dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:B.rows() 
                                                                       columns:B.cols() 
                                                                      rowBytes:B.cols() * sizeof(float) 
                                                                      dataType:MPSDataTypeFloat32];
    
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:A.rows() 
                                                                       columns:B.cols() 
                                                                      rowBytes:B.cols() * sizeof(float) 
                                                                      dataType:MPSDataTypeFloat32];
    
    // Create buffers
    id bufferA = createBufferFromMatrix(A);
    id bufferB = createBufferFromMatrix(B);
    id bufferC = [device_ newBufferWithLength:A.rows() * B.cols() * sizeof(float) 
                                      options:MTLResourceStorageModeShared];
    
    // Create MPS matrices
    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
    
    // Create multiplication kernel
    MPSMatrixMultiplication* multiplication = [[MPSMatrixMultiplication alloc] initWithDevice:device_ 
                                                                                 resultRows:A.rows() 
                                                                              resultColumns:B.cols() 
                                                                           interiorColumns:A.cols()];
    
    // Execute
    id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
    [multiplication encodeToCommandBuffer:commandBuffer 
                               leftMatrix:matrixA 
                              rightMatrix:matrixB 
                             resultMatrix:matrixC];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back
    C.resize(A.rows(), B.cols());
    copyMatrixFromBuffer(bufferC, C);
    
    // Cleanup
    [matrixA release];
    [matrixB release];
    [matrixC release];
    [multiplication release];
    [bufferA release];
    [bufferB release];
    [bufferC release];
}

void MetalDevice::matrixAdd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    if (!metalAvailable_) {
        C = A + B;
        return;
    }
    
    // For now, fallback to CPU for element-wise operations
    // Metal Performance Shaders doesn't have direct element-wise add for matrices
    C = A + B;
}

void MetalDevice::matrixSubtract(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    if (!metalAvailable_) {
        C = A - B;
        return;
    }
    
    // Fallback to CPU
    C = A - B;
}

void MetalDevice::matrixElementwiseMultiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    if (!metalAvailable_) {
        C = A.cwiseProduct(B);
        return;
    }
    
    // Fallback to CPU
    C = A.cwiseProduct(B);
}

void MetalDevice::matrixElementwiseDivide(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    if (!metalAvailable_) {
        C = A.cwiseQuotient(B);
        return;
    }
    
    // Fallback to CPU
    C = A.cwiseQuotient(B);
}

void MetalDevice::relu(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) {
    if (!metalAvailable_) {
        output = input.cwiseMax(0.0);
        return;
    }
    
    // Fallback to CPU for now
    output = input.cwiseMax(0.0);
}

void MetalDevice::reluBackward(const Eigen::MatrixXd& gradOutput, const Eigen::MatrixXd& input, Eigen::MatrixXd& gradInput) {
    if (!metalAvailable_) {
        gradInput = gradOutput.cwiseProduct((input.array() > 0.0).cast<double>().matrix());
        return;
    }
    
    // Fallback to CPU
    gradInput = gradOutput.cwiseProduct((input.array() > 0.0).cast<double>().matrix());
}

void MetalDevice::softmax(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) {
    if (!metalAvailable_) {
        // CPU softmax implementation
        output.resize(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); ++i) {
            Eigen::RowVectorXd row = input.row(i);
            double maxVal = row.maxCoeff();
            row = (row.array() - maxVal).exp();
            double sum = row.sum();
            output.row(i) = row / sum;
        }
        return;
    }
    
    // Fallback to CPU
    output.resize(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); ++i) {
        Eigen::RowVectorXd row = input.row(i);
        double maxVal = row.maxCoeff();
        row = (row.array() - maxVal).exp();
        double sum = row.sum();
        output.row(i) = row / sum;
    }
}

void MetalDevice::vectorAdd(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C) {
    if (!metalAvailable_) {
        C = A + B;
        return;
    }
    
    // Fallback to CPU
    C = A + B;
}

void MetalDevice::vectorSubtract(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C) {
    if (!metalAvailable_) {
        C = A - B;
        return;
    }
    
    // Fallback to CPU
    C = A - B;
}

void MetalDevice::vectorElementwiseMultiply(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C) {
    if (!metalAvailable_) {
        C = A.cwiseProduct(B);
        return;
    }
    
    // Fallback to CPU
    C = A.cwiseProduct(B);
}

void MetalDevice::vectorElementwiseDivide(const Eigen::RowVectorXd& A, const Eigen::RowVectorXd& B, Eigen::RowVectorXd& C) {
    if (!metalAvailable_) {
        C = A.cwiseQuotient(B);
        return;
    }
    
    // Fallback to CPU
    C = A.cwiseQuotient(B);
}

} // namespace Metal
} // namespace NEURAL_NETWORK

#endif // METAL_ENABLED
