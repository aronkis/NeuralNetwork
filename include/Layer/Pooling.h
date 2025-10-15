#ifndef __POOLING_H__
#define __POOLING_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK 
{
    class Pooling : public LayerBase
    {
    public:
        virtual ~Pooling() = default;

        void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override = 0;
        void backward(const Eigen::Tensor<double, 2>& dvalues) override = 0;

        const Eigen::Tensor<double, 2>& GetOutput() const override;
        const Eigen::Tensor<double, 2>& GetDInput() const override;
        void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;
        Eigen::Tensor<double, 2> predictions() const override;

        int GetPoolSize() const;
        int GetStride() const;
        int GetInputHeight() const;
        int GetInputWidth() const;
        int GetInputChannels() const;
        int GetOutputHeight() const;
        int GetOutputWidth() const;

    protected:
        Pooling(int batch_size, int pool_size, int input_height, 
                int input_width, int input_channels,
                int stride);

        void InputTensor2DToTensor4D(const Eigen::Tensor<double, 2>& tensor2d,
                                 int batch_size, int height,
                                 int width, int channels);
        Eigen::Tensor<double, 2> InputTensor4DToTensor2D(const Eigen::Tensor<double, 4>& tensor);

        Eigen::Tensor<double, 4> inputs_;
        Eigen::Tensor<double, 4> output_tensor_;
        Eigen::Tensor<double, 2> output_;

        Eigen::Tensor<double, 4> d_input_tensor_;
        Eigen::Tensor<double, 2> d_input_;

        int batch_size_;
        int pool_size_;
        int stride_;
        int input_height_;
        int input_width_;
        int input_channels_;

        int output_height_;
        int output_width_;
    };
} // namespace NEURAL_NETWORK

#endif // __POOLING_H__