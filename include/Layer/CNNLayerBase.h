#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "LayerInterface.h"

namespace NEURAL_NETWORK
{
    class CNNLayerBase : public LayerInterface
    {
    public:
        virtual ~CNNLayerBase() = default;

        // Pure tensor interface for CNN layers
        virtual void forward(const Eigen::Tensor<double, 4>& inputs, bool training) = 0;
        virtual void backward(const Eigen::Tensor<double, 4>& d_values) = 0;

        // Tensor getters
        virtual const Eigen::Tensor<double, 4>& GetTensorOutput() const = 0;
        virtual const Eigen::Tensor<double, 4>& GetTensorDInput() const = 0;
        virtual Eigen::Tensor<double, 4> tensorPredictions() const = 0;
    };
}