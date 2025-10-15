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

    // Initialize tensors
    inputs_ = Eigen::Tensor<double, 4>(batch_size_, input_height_, input_width_, input_channels_);
    d_input_tensor_ = Eigen::Tensor<double, 4>(batch_size_, input_height_, input_width_, input_channels_);
}

void NEURAL_NETWORK::Pooling::InputTensor2DToTensor4D(const Eigen::Tensor<double, 2>& tensor2d,
                                                    int batch_size, int height,
                                                    int width, int channels)
{
    // Convert tensor2d to inputs_ tensor4d manually
    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = c * (height * width) + h * width + w;
                    inputs_(b, h, w, c) = tensor2d(b, idx);
                }
            }
        }
    }
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Pooling::InputTensor4DToTensor2D(const Eigen::Tensor<double, 4>& tensor)
{
    int batch_size = tensor.dimension(0);
    int height = tensor.dimension(1);
    int width = tensor.dimension(2);
    int channels = tensor.dimension(3);

    Eigen::Tensor<double, 2> tensor2d(batch_size, height * width * channels);

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = c * (height * width) + h * width + w;
                    tensor2d(b, idx) = tensor(b, h, w, c);
                }
            }
        }
    }

    return tensor2d;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Pooling::GetOutput() const
{
    return output_;
}

const Eigen::Tensor<double, 2>& NEURAL_NETWORK::Pooling::GetDInput() const
{
    return d_input_;
}

void NEURAL_NETWORK::Pooling::SetDInput(const Eigen::Tensor<double, 2>& dinput)
{
    d_input_ = dinput;
}

Eigen::Tensor<double, 2> NEURAL_NETWORK::Pooling::predictions() const
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
