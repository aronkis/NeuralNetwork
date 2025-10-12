#ifndef __MAX_POOLING_H__
#define __MAX_POOLING_H__

#include "Pooling.h"

namespace NEURAL_NETWORK
{
    class MaxPooling : public Pooling
    {
    public:
        MaxPooling(int batch_size, int pool_size, int input_height, 
                   int input_width, int input_channels, 
                   int stride);
        ~MaxPooling() = default;

        void forward(const Eigen::MatrixXd& inputs, bool training) override;
        void backward(const Eigen::MatrixXd& dvalues) override;
    private:
        Eigen::Tensor<int, 4> max_indices_;
    };
} // namespace NEURAL_NETWORK

#endif // __POOLING_2D_H__

