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

        void forward(const Eigen::MatrixXd& inputs, bool training) override;
        void backward(const Eigen::MatrixXd& dvalues) override;
        const Eigen::MatrixXd& GetOutput() const override;
        const Eigen::MatrixXd& GetDInput() const override;
        void SetDInput(const Eigen::MatrixXd& dinput) override;
        Eigen::MatrixXd predictions() const override;

        void storeTargets(const Eigen::MatrixXi& targets);

    private:
        ActivationSoftmax softmax_;

        Eigen::MatrixXd output_;
        Eigen::MatrixXd d_inputs_;
        Eigen::MatrixXi targets_;
    };

} // namespace NEURAL_NETWORK

#endif // __ACTIVATION_SOFTMAX_LOSS_CATEGORICAL_CROSSENTROPY_H__