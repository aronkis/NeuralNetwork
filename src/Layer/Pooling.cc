#include "Pooling.h"
#include "TensorUtils.h"

NEURAL_NETWORK::Pooling::Pooling(int batch_size, int pool_size, int input_height, 
                                 int input_width, int input_channels,
                                 int stride)
{
    batch_size_ = batch_size;
    pool_size_ = pool_size;
    stride_ = stride;
    input_height_ = input_height;
    input_width_ = input_width;
    input_channels_ = input_channels;

    output_height_ = 1 + (input_height - pool_size) / stride;
    output_width_ = 1 + (input_width - pool_size) / stride;
}

void NEURAL_NETWORK::Pooling::InputMatrixToTensor(const Eigen::MatrixXd& matrix, 
                                                  int batch_size, int height, 
                                                  int width, int channels)
{
    TensorUtils::MatrixToTensor4D(matrix, inputs_, batch_size, height, width, channels);
}

Eigen::MatrixXd NEURAL_NETWORK::Pooling::InputTensorToMatrix(const Eigen::Tensor<double, 4>& tensor)
{
    return TensorUtils::Tensor4DToMatrix(tensor);
}

const Eigen::MatrixXd& NEURAL_NETWORK::Pooling::GetOutput() const
{
    return output_;
}
const Eigen::MatrixXd& NEURAL_NETWORK::Pooling::GetDInput() const
{
    return d_input_;
}
void NEURAL_NETWORK::Pooling::SetDInput(const Eigen::MatrixXd& d_input)
{
    d_input_ = d_input;
}

Eigen::MatrixXd NEURAL_NETWORK::Pooling::predictions() const
{
    return output_;
}

int NEURAL_NETWORK::Pooling::GetPoolSize() const
{
    return pool_size_;
}

int NEURAL_NETWORK::Pooling::GetStride() const
{
    return stride_;
}

int NEURAL_NETWORK::Pooling::GetInputHeight() const
{
    return input_height_;
}

int NEURAL_NETWORK::Pooling::GetInputWidth() const
{
    return input_width_;
}

int NEURAL_NETWORK::Pooling::GetInputChannels() const
{
    return input_channels_;
}

int NEURAL_NETWORK::Pooling::GetOutputHeight() const
{
    return output_height_;
}

int NEURAL_NETWORK::Pooling::GetOutputWidth() const
{
    return output_width_;
}

// Tensor interface implementations
bool NEURAL_NETWORK::Pooling::SupportsTensorInterface() const
{
    return true;
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Pooling::GetTensorOutput() const
{
    return output_tensor_;
}

const Eigen::Tensor<double, 4>& NEURAL_NETWORK::Pooling::GetTensorDInput() const
{
    return d_input_tensor_;
}

void NEURAL_NETWORK::Pooling::SetTensorDInput(const Eigen::Tensor<double, 4>& dinput)
{
    d_input_tensor_ = dinput;
    // Also update matrix version for backward compatibility
    d_input_ = InputTensorToMatrix(d_input_tensor_);
}
