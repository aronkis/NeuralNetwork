#pragma once

#include "CNNLayerBase.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace NEURAL_NETWORK
{
    class TrainableCNNLayer : public CNNLayerBase
    {
    public:
        virtual ~TrainableCNNLayer() = default;

        // Trainable layer interface for CNN layers using tensors
        virtual const Eigen::Tensor<double, 4>& GetWeights() const = 0;
        virtual const Eigen::Tensor<double, 1>& GetBiases() const = 0;
        virtual const Eigen::Tensor<double, 4>& GetDWeights() const = 0;
        virtual const Eigen::Tensor<double, 1>& GetDBiases() const = 0;

        // Optimizer state for CNN layers
        virtual const Eigen::Tensor<double, 4>& GetWeightMomentums() const = 0;
        virtual const Eigen::Tensor<double, 1>& GetBiasMomentums() const = 0;
        virtual const Eigen::Tensor<double, 4>& GetWeightCaches() const = 0;
        virtual const Eigen::Tensor<double, 1>& GetBiasCaches() const = 0;

        virtual void SetWeightMomentums(const Eigen::Tensor<double, 4>& momentums) = 0;
        virtual void SetBiasMomentums(const Eigen::Tensor<double, 1>& momentums) = 0;
        virtual void SetWeightCaches(const Eigen::Tensor<double, 4>& caches) = 0;
        virtual void SetBiasCaches(const Eigen::Tensor<double, 1>& caches) = 0;

        virtual void UpdateWeights(const Eigen::Tensor<double, 4>& update) = 0;
        virtual void UpdateBiases(const Eigen::Tensor<double, 1>& update) = 0;

        // Parameter management
        virtual std::pair<Eigen::Tensor<double, 4>, Eigen::Tensor<double, 1>> GetTensorParameters() const = 0;
        virtual void SetTensorParameters(const Eigen::Tensor<double, 4>& weights, const Eigen::Tensor<double, 1>& biases) = 0;

        // Regularization
        virtual double GetWeightRegularizerL1() const { return 0.0; }
        virtual double GetWeightRegularizerL2() const { return 0.0; }
        virtual double GetBiasRegularizerL1() const { return 0.0; }
        virtual double GetBiasRegularizerL2() const { return 0.0; }

        bool isTrainable() const override { return true; }
    };
}