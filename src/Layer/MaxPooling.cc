#include "MaxPooling.h"
#include "TensorUtils.h"

NEURAL_NETWORK::MaxPooling::MaxPooling(int batch_size, int pool_size, int input_height, 
                                       int input_width, int input_channels, 
                                       int stride)
    : Pooling(batch_size, pool_size, input_height, input_width, input_channels, stride)
{
    max_indices_ = Eigen::Tensor<int, 4>(batch_size, output_height_, output_width_, input_channels_);
    output_tensor_ = Eigen::Tensor<double, 4>(batch_size, output_height_, output_width_, input_channels_);
}

void NEURAL_NETWORK::MaxPooling::forward(const Eigen::MatrixXd& inputs, bool training)
{
    int actual_batch_size = inputs.rows();
    
    if (actual_batch_size != batch_size_)
    {
        batch_size_ = actual_batch_size;
        max_indices_.resize(batch_size_, output_height_, output_width_, input_channels_);
        output_tensor_.resize(batch_size_, output_height_, output_width_, input_channels_);
        inputs_.resize(batch_size_, input_height_, input_width_, input_channels_);
    }
    
    InputMatrixToTensor(inputs, batch_size_, input_height_, input_width_, input_channels_);

    Eigen::MatrixXd col = TensorUtils::im2col(inputs_, 
                                              pool_size_, pool_size_,
                                              0, 0,
                                              stride_, stride_);
    
    int window_size = pool_size_ * pool_size_ * input_channels_;
    int num_windows = batch_size_ * output_height_ * output_width_;
    
    for (int w = 0; w < num_windows; w++)
    {
        Eigen::VectorXd window_col = col.col(w);
        
        // Convert flat window index to 4D coordinates
        int b = w / (output_height_ * output_width_);
        int remaining = w % (output_height_ * output_width_);
        int h = remaining / output_width_;
        int ow = remaining % output_width_;
        
        for (int c = 0; c < input_channels_; c++)
        {
            int start_idx = c * (pool_size_ * pool_size_);
            int end_idx = start_idx + (pool_size_ * pool_size_);
            
            Eigen::VectorXd channel_window = window_col.segment(start_idx, pool_size_ * pool_size_);
            
            Eigen::Index max_idx;
            double max_val = channel_window.maxCoeff(&max_idx);
            
            output_tensor_(b, h, ow, c) = max_val;
            max_indices_(b, h, ow, c) = static_cast<int>(max_idx);
        }
    }
    
    output_ = TensorUtils::Tensor4DToMatrix(output_tensor_);
}

void NEURAL_NETWORK::MaxPooling::backward(const Eigen::MatrixXd& dvalues)
{
    int actual_batch_size = dvalues.rows();
    
    if (d_input_tensor_.dimension(0) != actual_batch_size)
    {
        d_input_tensor_.resize(actual_batch_size, input_height_, input_width_, input_channels_);
    }
    
    Eigen::Tensor<double, 4> dvalues_tensor(actual_batch_size, output_height_, output_width_, input_channels_);
    TensorUtils::MatrixToTensor4D(dvalues, dvalues_tensor, actual_batch_size, output_height_, output_width_, input_channels_);
    
    int window_size = pool_size_ * pool_size_ * input_channels_;
    int num_windows = actual_batch_size * output_height_ * output_width_;
    Eigen::MatrixXd d_col = Eigen::MatrixXd::Zero(window_size, num_windows);
    
    for (int w = 0; w < num_windows; w++) 
    {
        int b = w / (output_height_ * output_width_);
        int remaining = w % (output_height_ * output_width_);
        int h = remaining / output_width_;
        int ow = remaining % output_width_;
        
        for (int c = 0; c < input_channels_; c++)
        {
            int max_idx = max_indices_(b, h, ow, c);
            
            int d_col_idx = c * (pool_size_ * pool_size_) + max_idx;
            
            d_col(d_col_idx, w) = dvalues_tensor(b, h, ow, c);
        }
    }
    
    TensorUtils::col2im(d_col, d_input_tensor_,
                       actual_batch_size, input_height_, input_width_, input_channels_,
                       pool_size_, pool_size_,
                       0, 0,
                       stride_, stride_);
    
    d_input_ = TensorUtils::Tensor4DToMatrix(d_input_tensor_);
}