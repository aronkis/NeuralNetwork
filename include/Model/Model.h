#ifndef __MODEL_H__
#define __MODEL_H__

#include "NeuralNetwork.h"
#include <memory>
#include <string>

namespace NEURAL_NETWORK 
{
    class Model
    {
    public:
        Model() = default;
        ~Model() = default;
        
        Model(const Model&) = delete;
        Model& operator=(const Model&) = delete;

        void Add(std::shared_ptr<LayerBase> layer);
        void Set(std::unique_ptr<Loss> loss, 
				 std::unique_ptr<Accuracy> accuracy, 
				 std::unique_ptr<Optimizer> optimizer = nullptr);
        void Finalize();
        void Train(const Eigen::MatrixXd& X, 
				   const Eigen::MatrixXd& y,
				   int batch_size, int epochs, int print_every, 
				   const Eigen::MatrixXd& X_val, 
				   const Eigen::MatrixXd& y_val);
		void Evaluate(const Eigen::MatrixXd& X, 
					  const Eigen::MatrixXd& y, 
					  int batch_size);
		Eigen::MatrixXd Predict(const Eigen::MatrixXd& X, int batch_size);

		std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> GetParameters() const;
		void SaveParameters(const std::string& path) const;
		void SaveModel(const std::string& path) const;

		void SetParameters(const std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>>& params);
		void LoadParameters(const std::string& path);
		void LoadModel(const std::string& path);

    private:
        void forward(const Eigen::MatrixXd& inputs, bool training);
        void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& targets);

		std::shared_ptr<LayerInput> input_layer_;
		std::vector<std::shared_ptr<LayerBase>> layers_;
    	std::vector<std::shared_ptr<LayerDense>> trainable_layers_;
        std::unique_ptr<ActivationSoftmaxLossCategoricalCrossEntropy> softmax_loss_;
        std::unique_ptr<Loss> loss_;
        std::unique_ptr<Accuracy> accuracy_;
        std::unique_ptr<Optimizer> optimizer_;

        Eigen::MatrixXd output_;
        
        bool softmax_classifier_ = false;
    };
} // namespace NEURAL_NETWORK

#endif // __MODEL_H__