#ifndef __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__
#define __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__

#include "ActivationSoftmax.h"
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
    class ActivationSoftmaxLossCategoricalCrossEntropy : public LayerBase
    {
    public:
        ActivationSoftmaxLossCategoricalCrossEntropy() = default;
        ~ActivationSoftmaxLossCategoricalCrossEntropy() = default;

        ActivationSoftmaxLossCategoricalCrossEntropy(const ActivationSoftmaxLossCategoricalCrossEntropy&) = delete;
	    ActivationSoftmaxLossCategoricalCrossEntropy& operator=(const ActivationSoftmaxLossCategoricalCrossEntropy&) = delete;

        void forward(const Eigen::Tensor<double, 2>& inputs, bool training) override;
        void backward(const Eigen::Tensor<double, 2>& dvalues) override;
        Eigen::Tensor<double, 2> predictions() const override;
        void storeTargets(const Eigen::Tensor<int, 2>& targets);

        const Eigen::Tensor<double, 2>& GetOutput() const override;
        const Eigen::Tensor<double, 2>& GetDInput() const override;

		void SetDInput(const Eigen::Tensor<double, 2>& dinput) override;
    private:
        ActivationSoftmax softmax_;

        Eigen::Tensor<double, 2> output_;
        Eigen::Tensor<double, 2> d_inputs_;
        Eigen::Tensor<int, 2> targets_;
    };
} // namespace NEURAL_NETWORK

#endif // __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__