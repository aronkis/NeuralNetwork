#ifndef __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "Activation/ActivationSoftmax.h"
#include "Loss/LossCategoricalCrossentropy.h"

namespace NEURAL_NETWORK
{

    class ActivationSoftmaxLossCategoricalCrossentropy
    {
    public:
        ActivationSoftmaxLossCategoricalCrossentropy() = default;
        ~ActivationSoftmaxLossCategoricalCrossentropy() = default;

        ActivationSoftmaxLossCategoricalCrossentropy(const ActivationSoftmaxLossCategoricalCrossentropy&) = delete;
		ActivationSoftmaxLossCategoricalCrossentropy& operator=(const ActivationSoftmaxLossCategoricalCrossentropy&) = delete;

        void forward(const Eigen::MatrixXd& inputs,
                     const Eigen::MatrixXi& targets);

        void backward(const Eigen::MatrixXd& d_values,
                      const Eigen::MatrixXi& targets);

        double GetLoss() const;
        const Eigen::MatrixXd& GetOutput() const;
        const Eigen::MatrixXd& GetDInputs() const;

    private:
        ActivationSoftmax softmax_;
        LossCategoricalCrossentropy loss_;

        double loss_value_ = 0.0;
        Eigen::MatrixXd output_;

        Eigen::MatrixXd d_inputs_;
    };

} // namespace NEURAL_NETWORK

#endif // __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__