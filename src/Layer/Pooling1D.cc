#include "Pooling1D.h"

NEURAL_NETWORK::Pooling1D::Pooling1D(int batch_size, int pool_size, int input_length,
                                      int input_channels, int stride)
{
    batch_size_ = batch_size;
    pool_size_ = pool_size;
    input_length_ = input_length;
    input_channels_ = input_channels;
    stride_ = stride;
    output_length_ = (input_length - pool_size) / stride + 1;

    inputs_ = Eigen::Tensor<double, 3>(batch_size_, input_length_, input_channels_);
    output_tensor_ = Eigen::Tensor<double, 3>(batch_size_, output_length_, input_channels_);
    d_input_tensor_ = Eigen::Tensor<double, 3>(batch_size_, input_length_, input_channels_);
}

const Eigen::MatrixXd& NEURAL_NETWORK::Pooling1D::GetOutput() const
{
    return output_;
}

const Eigen::MatrixXd& NEURAL_NETWORK::Pooling1D::GetDInput() const
{
    return d_input_;
}

void NEURAL_NETWORK::Pooling1D::SetDInput(const Eigen::MatrixXd& dinput)
{
    d_input_ = dinput;
}

Eigen::MatrixXd NEURAL_NETWORK::Pooling1D::predictions() const
{
    return output_;
}

int NEURAL_NETWORK::Pooling1D::GetPoolSize() const
{
    return pool_size_;
}

int NEURAL_NETWORK::Pooling1D::GetStride() const
{
    return stride_;
}

int NEURAL_NETWORK::Pooling1D::GetInputLength() const
{
    return input_length_;
}

int NEURAL_NETWORK::Pooling1D::GetInputChannels() const
{
    return input_channels_;
}

int NEURAL_NETWORK::Pooling1D::GetOutputLength() const
{
    return output_length_;
}

void NEURAL_NETWORK::Pooling1D::InputMatrixToTensor(const Eigen::MatrixXd& matrix,
                                                     int batch_size, int length, int channels)
{
    if (inputs_.dimension(0) != batch_size)
    {
        inputs_.resize(batch_size, length, channels);
        output_tensor_.resize(batch_size, output_length_, channels);
        d_input_tensor_.resize(batch_size, length, channels);
        batch_size_ = batch_size;
    }

    for (int b = 0; b < batch_size; b++)
    {
        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < channels; c++)
            {
                int col = t * channels + c;
                inputs_(b, t, c) = matrix(b, col);
            }
        }
    }
}

Eigen::MatrixXd NEURAL_NETWORK::Pooling1D::InputTensorToMatrix(const Eigen::Tensor<double, 3>& tensor)
{
    int batch_size = static_cast<int>(tensor.dimension(0));
    int length = static_cast<int>(tensor.dimension(1));
    int channels = static_cast<int>(tensor.dimension(2));

    Eigen::MatrixXd matrix(batch_size, length * channels);
    for (int b = 0; b < batch_size; b++)
    {
        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < channels; c++)
            {
                int col = t * channels + c;
                matrix(b, col) = tensor(b, t, c);
            }
        }
    }
    return matrix;
}