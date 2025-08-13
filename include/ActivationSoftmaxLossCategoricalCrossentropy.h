#ifndef __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "ActivationSoftmax.h"
#include "LossCategoricalCrossEntropy.h"

namespace NEURAL_NETWORK
{
    class ActivationSoftmaxLossCategoricalCrossEntropy
    {
    public:
        ActivationSoftmaxLossCategoricalCrossEntropy() = default;
        ~ActivationSoftmaxLossCategoricalCrossEntropy() = default;

        ActivationSoftmaxLossCategoricalCrossEntropy(const ActivationSoftmaxLossCategoricalCrossEntropy&) = delete;
	    ActivationSoftmaxLossCategoricalCrossEntropy& operator=(const ActivationSoftmaxLossCategoricalCrossEntropy&) = delete;

        void forward(const Eigen::MatrixXd& inputs,
                     const Eigen::MatrixXi& targets);

        void backward(const Eigen::MatrixXd& d_values,
                      const Eigen::MatrixXi& targets);

        double GetLoss() const;
        const Eigen::MatrixXd& GetOutput() const;
        const Eigen::MatrixXd& GetDInput() const;

        LossCategoricalCrossEntropy& GetLossFunction();

    private:
        ActivationSoftmax softmax_;
        LossCategoricalCrossEntropy loss_;

        double loss_value_ = 0.0;
        Eigen::MatrixXd output_;

        Eigen::MatrixXd d_inputs_;
    };

} // namespace NEURAL_NETWORK

#endif // __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__