#ifndef __POOLING_1D_H__
#define __POOLING_1D_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LayerBase.h"

namespace NEURAL_NETWORK
{
    class Pooling1D : public LayerBase
    {
    public:
        virtual ~Pooling1D() = default;

        void forward(const Eigen::MatrixXd& inputs, bool training) override = 0;
        void backward(const Eigen::MatrixXd& dvalues) override = 0;

        const Eigen::MatrixXd& GetOutput() const override;
        const Eigen::MatrixXd& GetDInput() const override;
        void SetDInput(const Eigen::MatrixXd& dinput) override;
        Eigen::MatrixXd predictions() const override;

        // 1D-specific getters
        int GetPoolSize() const;
        int GetStride() const;
        int GetInputLength() const;
        int GetInputChannels() const;
        int GetOutputLength() const;

        // LayerBase required methods (no weights/biases in pooling)
        double GetWeightRegularizerL1() const override { return 0.0; }
        double GetWeightRegularizerL2() const override { return 0.0; }
        double GetBiasRegularizerL1() const override { return 0.0; }
        double GetBiasRegularizerL2() const override { return 0.0; }

        const Eigen::MatrixXd& GetWeights() const override {
            static Eigen::MatrixXd empty; return empty;
        }
        const Eigen::RowVectorXd& GetBiases() const override {
            static Eigen::RowVectorXd empty; return empty;
        }

        std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> GetParameters() const override {
            return {GetWeights(), GetBiases()};
        }
        void SetParameters(const Eigen::MatrixXd& weights, const Eigen::RowVectorXd& biases) override {}

        const Eigen::MatrixXd& GetDWeights() const override {
            static Eigen::MatrixXd empty; return empty;
        }
        const Eigen::RowVectorXd& GetDBiases() const override {
            static Eigen::RowVectorXd empty; return empty;
        }

        const Eigen::MatrixXd& GetWeightMomentums() const override {
            static Eigen::MatrixXd empty; return empty;
        }
        const Eigen::RowVectorXd& GetBiasMomentums() const override {
            static Eigen::RowVectorXd empty; return empty;
        }
        const Eigen::MatrixXd& GetWeightCaches() const override {
            static Eigen::MatrixXd empty; return empty;
        }
        const Eigen::RowVectorXd& GetBiasCaches() const override {
            static Eigen::RowVectorXd empty; return empty;
        }

        void SetWeightMomentums(const Eigen::MatrixXd& weight_momentums) override {}
        void SetBiasMomentums(const Eigen::RowVectorXd& bias_momentums) override {}
        void SetWeightCaches(const Eigen::MatrixXd& weight_caches) override {}
        void SetBiasCaches(const Eigen::RowVectorXd& bias_caches) override {}

        void UpdateWeights(Eigen::MatrixXd& weight_update) override {}
        void UpdateWeightsCache(Eigen::MatrixXd& weight_update) override {}
        void UpdateBiases(Eigen::RowVectorXd& bias_update) override {}
        void UpdateBiasesCache(Eigen::RowVectorXd& bias_update) override {}

    protected:
        Pooling1D(int batch_size, int pool_size, int input_length,
                  int input_channels, int stride);

        void InputMatrixToTensor(const Eigen::MatrixXd& matrix,
                                 int batch_size, int length, int channels);
        Eigen::MatrixXd InputTensorToMatrix(const Eigen::Tensor<double, 3>& tensor);

        Eigen::Tensor<double, 3> inputs_;
        Eigen::Tensor<double, 3> output_tensor_;
        Eigen::MatrixXd output_;

        Eigen::Tensor<double, 3> d_input_tensor_;
        Eigen::MatrixXd d_input_;

        int batch_size_;
        int pool_size_;
        int stride_;
        int input_length_;
        int input_channels_;

        int output_length_;
    };
} // namespace NEURAL_NETWORK

#endif // __POOLING_1D_H__