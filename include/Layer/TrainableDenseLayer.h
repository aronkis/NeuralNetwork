#pragma once

#include "DenseLayerBase.h"
#include <Eigen/Dense>

namespace NEURAL_NETWORK
{
    class TrainableDenseLayer : public DenseLayerBase
    {
    public:
        virtual ~TrainableDenseLayer() = default;

        // Trainable layer interface for dense layers using matrices
        virtual const Eigen::MatrixXd& GetWeights() const = 0;
        virtual const Eigen::RowVectorXd& GetBiases() const = 0;
        virtual const Eigen::MatrixXd& GetDWeights() const = 0;
        virtual const Eigen::RowVectorXd& GetDBiases() const = 0;

        // Optimizer state for dense layers
        virtual const Eigen::MatrixXd& GetWeightMomentums() const = 0;
        virtual const Eigen::RowVectorXd& GetBiasMomentums() const = 0;
        virtual const Eigen::MatrixXd& GetWeightCaches() const = 0;
        virtual const Eigen::RowVectorXd& GetBiasCaches() const = 0;

        virtual void SetWeightMomentums(const Eigen::MatrixXd& momentums) = 0;
        virtual void SetBiasMomentums(const Eigen::RowVectorXd& momentums) = 0;
        virtual void SetWeightCaches(const Eigen::MatrixXd& caches) = 0;
        virtual void SetBiasCaches(const Eigen::RowVectorXd& caches) = 0;

        virtual void UpdateWeights(Eigen::MatrixXd& update) = 0;
        virtual void UpdateBiases(Eigen::RowVectorXd& update) = 0;
        virtual void UpdateWeightsCache(Eigen::MatrixXd& update) = 0;
        virtual void UpdateBiasesCache(Eigen::RowVectorXd& update) = 0;

        // Parameter management
        virtual std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const = 0;
        virtual void SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases) = 0;

        // Regularization
        virtual double GetWeightRegularizerL1() const { return 0.0; }
        virtual double GetWeightRegularizerL2() const { return 0.0; }
        virtual double GetBiasRegularizerL1() const { return 0.0; }
        virtual double GetBiasRegularizerL2() const { return 0.0; }

        bool isTrainable() const override { return true; }
    };
}