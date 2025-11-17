#include "MaxPooling1D.h"

NEURAL_NETWORK::MaxPooling1D::MaxPooling1D(int batch_size, int pool_size, int input_length,
                                            int input_channels, int stride)
    : Pooling1D(batch_size, pool_size, input_length, input_channels, stride)
{
    max_indices_ = Eigen::Tensor<int, 3>(batch_size_, output_length_, input_channels_);
}

void NEURAL_NETWORK::MaxPooling1D::forward(const Eigen::MatrixXd& inputs, bool training)
{
    int actual_batch_size = inputs.rows();

    if (actual_batch_size != batch_size_)
    {
        batch_size_ = actual_batch_size;
        max_indices_.resize(batch_size_, output_length_, input_channels_);
        output_tensor_.resize(batch_size_, output_length_, input_channels_);
        inputs_.resize(batch_size_, input_length_, input_channels_);
    }

    InputMatrixToTensor(inputs, batch_size_, input_length_, input_channels_);

    for (int b = 0; b < batch_size_; b++)
    {
        for (int c = 0; c < input_channels_; c++)
        {
            for (int out_t = 0; out_t < output_length_; out_t++)
            {
                int start_t = out_t * stride_;
                int end_t = std::min(start_t + pool_size_, input_length_);

                double max_val = inputs_(b, start_t, c);
                int max_idx = 0;

                for (int t = start_t; t < end_t; t++)
                {
                    if (inputs_(b, t, c) > max_val)
                    {
                        max_val = inputs_(b, t, c);
                        max_idx = t - start_t;
                    }
                }

                output_tensor_(b, out_t, c) = max_val;
                max_indices_(b, out_t, c) = max_idx;
            }
        }
    }

    output_ = InputTensorToMatrix(output_tensor_);
}

void NEURAL_NETWORK::MaxPooling1D::backward(const Eigen::MatrixXd& dvalues)
{
    int actual_batch_size = dvalues.rows();

    if (d_input_tensor_.dimension(0) != actual_batch_size)
    {
        d_input_tensor_.resize(actual_batch_size, input_length_, input_channels_);
    }

    d_input_tensor_.setZero();

    Eigen::Tensor<double, 3> dvalues_tensor(actual_batch_size, output_length_, input_channels_);
    for (int b = 0; b < actual_batch_size; b++)
    {
        for (int out_t = 0; out_t < output_length_; out_t++)
        {
            for (int c = 0; c < input_channels_; c++)
            {
                int col = out_t * input_channels_ + c;
                dvalues_tensor(b, out_t, c) = dvalues(b, col);
            }
        }
    }

    for (int b = 0; b < actual_batch_size; b++)
    {
        for (int c = 0; c < input_channels_; c++)
        {
            for (int out_t = 0; out_t < output_length_; out_t++)
            {
                int start_t = out_t * stride_;
                int max_idx = max_indices_(b, out_t, c);
                int actual_input_idx = start_t + max_idx;

                if (actual_input_idx < input_length_)
                {
                    d_input_tensor_(b, actual_input_idx, c) += dvalues_tensor(b, out_t, c);
                }
            }
        }
    }

    d_input_ = InputTensorToMatrix(d_input_tensor_);
}