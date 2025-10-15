#pragma once

#include <Eigen/Dense>
#include <memory>
#include "LayerInterface.h"

namespace NEURAL_NETWORK
{
    class DenseLayerBase : public LayerInterface
    {
    public:
        virtual ~DenseLayerBase() = default;

        // Pure matrix interface for dense layers
        virtual void forward(const Eigen::MatrixXd& inputs, bool training) = 0;
        virtual void backward(const Eigen::MatrixXd& d_values) = 0;

        // Matrix getters
        virtual const Eigen::MatrixXd& GetOutput() const = 0;
        virtual const Eigen::MatrixXd& GetDInput() const = 0;
        virtual Eigen::MatrixXd predictions() const = 0;

        // For backward compatibility
        virtual void SetDInput(const Eigen::MatrixXd& dinput) = 0;
    };
}