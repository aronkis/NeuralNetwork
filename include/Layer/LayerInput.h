#ifndef __LAYER_INPUT_H__
#define __LAYER_INPUT_H__

#include <Eigen/Dense>
#include "LayerBase.h"

namespace NEURAL_NETWORK 
{

    class LayerInput : public LayerBase
    {
    public:
        LayerInput() = default;
        ~LayerInput() = default;

        void forward(const Eigen::MatrixXd& inputs, bool training) override;
        void backward(const Eigen::MatrixXd& dvalues) override;
        const Eigen::MatrixXd& GetOutput() const override;
        const Eigen::MatrixXd& GetDInput() const override;
        Eigen::MatrixXd predictions() const override;
        
        void SetDInput(const Eigen::MatrixXd& dinput) override {}

    private:
        Eigen::MatrixXd output_;
    };
} // namespace NEURAL_NETWORK

#endif // __LAYER_INPUT_H__