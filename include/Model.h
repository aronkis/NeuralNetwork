#ifndef __MODEL_H__
#define __MODEL_H__

#include "LayerDense.h"
#include "LayerBase.h"
#include "LayerInput.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Accuracy.h"
#include <memory>

namespace NEURAL_NETWORK {

    class Model
    {
    public:
        Model() = default;
        ~Model() = default;
        
        Model(const Model&) = delete;
        Model& operator=(const Model&) = delete;

        void add(std::unique_ptr<LayerBase> layer);
        void set(std::unique_ptr<Loss> loss, std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Accuracy> accuracy);
        void finalize();
        void train(const Eigen::MatrixXd& X, Eigen::MatrixXd& y, int epochs, 
                    int print_every, const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& validation_data,
                    bool plot_validation_data);
    private:
        void forward(const Eigen::MatrixXd& inputs, bool training);
        void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& targets);

        std::unique_ptr<LayerInput> input_layer_;
        
        std::vector<std::unique_ptr<LayerBase>> layers_;
        std::vector<LayerDense*> trainable_layers_; 

        Eigen::MatrixXd output_;

        std::unique_ptr<Loss> loss_;
        std::unique_ptr<Optimizer> optimizer_;
        std::unique_ptr<Accuracy> accuracy_;
        std::unique_ptr<ActivationSoftmaxLossCategoricalCrossEntropy> softmax_loss_;
        
        bool softmax_classifier = false;
    };
} // namespace NEURAL_NETWORK

#endif // __MODEL_H__