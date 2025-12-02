#ifndef __MAX_POOLING_1D_H__
#define __MAX_POOLING_1D_H__

#include "Pooling1D.h"

namespace NEURAL_NETWORK
{
    class MaxPooling1D : public Pooling1D
    {
    public:
        MaxPooling1D(int batch_size, int pool_size, int input_length,
                     int input_channels, int stride = 1);
        ~MaxPooling1D() = default;

        void forward(const Eigen::MatrixXd& inputs, bool training) override;
        void backward(const Eigen::MatrixXd& dvalues) override;

    private:
        Eigen::Tensor<int, 3> max_indices_;
    };
} // namespace NEURAL_NETWORK

#endif // __MAX_POOLING_1D_H__