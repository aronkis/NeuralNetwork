#ifndef __MODEL_H__
#define __MODEL_H__

#include "NeuralNetwork.h"
#include "Helpers.h"
#include <memory>
#include <string>
#include <type_traits>

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
		// Tensor-based implementations for 2D data (Dense layers)
		void TrainTensor2D(const Eigen::Tensor<double, 2>& X, const Eigen::Tensor<double, 2>& y,
						   int batch_size, int epochs, int print_every,
						   const Eigen::Tensor<double, 2>& X_val, const Eigen::Tensor<double, 2>& y_val);

		void EvaluateTensor2D(const Eigen::Tensor<double, 2>& X, const Eigen::Tensor<double, 2>& y, int batch_size);

		Eigen::Tensor<double, 2> PredictTensor2D(const Eigen::Tensor<double, 2>& X, int batch_size);

		// Tensor-based implementations for 4D data (CNN layers)
		void TrainTensor4D(const Eigen::Tensor<double, 4>& X_tensor, const Eigen::Tensor<double, 2>& y_tensor,
						   int batch_size, int epochs, int print_every,
						   const Eigen::Tensor<double, 4>& X_val_tensor, const Eigen::Tensor<double, 2>& y_val_tensor);

		void EvaluateTensor4D(const Eigen::Tensor<double, 4>& X_tensor, const Eigen::Tensor<double, 2>& y_tensor, int batch_size);

		Eigen::Tensor<double, 2> PredictTensor4D(const Eigen::Tensor<double, 4>& X_tensor, int batch_size);

		// Template-based implementations that work with both tensor types
		template<typename XType, typename YType>
		void Train(const XType& X, const YType& y,
				   int batch_size, int epochs, int print_every,
				   const XType& X_val, const YType& y_val);

		template<typename XType, typename YType>
		void Evaluate(const XType& X, const YType& y, int batch_size);

		template<typename XType>
		Eigen::Tensor<double, 2> Predict(const XType& X, int batch_size);

		std::vector<std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>>> GetParameters() const;
		void SaveParameters(const std::string& path) const;
		void SaveModel(const std::string& path) const;

		void SetParameters(const std::vector<std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<double, 1>>>& params);
		void LoadParameters(const std::string& path);
		void LoadModel(const std::string& path);

    private:
		void forwardTensor2D(const Eigen::Tensor<double, 2>& inputs, bool training);
		void forwardTensor4D(const Eigen::Tensor<double, 4>& inputs, bool training);

		template<typename InputType>
        void forward(const InputType& inputs, bool training);
        void backward(const Eigen::Tensor<double, 2>& output, const Eigen::Tensor<double, 2>& targets);

		std::shared_ptr<LayerInput> input_layer_;
		std::vector<std::shared_ptr<LayerBase>> layers_;
    	std::vector<std::shared_ptr<LayerBase>> trainable_layers_;
        std::unique_ptr<ActivationSoftmaxLossCategoricalCrossEntropy> softmax_loss_;
        std::unique_ptr<Loss> loss_;
        std::unique_ptr<Accuracy> accuracy_;
        std::unique_ptr<Optimizer> optimizer_;

        Eigen::Tensor<double, 2> output_;

        bool softmax_classifier_ = false;
    };

    // Template method declarations (implementations in Model.cc)
    // Explicit instantiations will be provided for:
    // - Eigen::MatrixXd (backward compatibility)
    // - Eigen::Tensor<double, 4> (CNN data)
    // - Eigen::Tensor<double, 2> (labels/Dense layer data)

} // namespace NEURAL_NETWORK

#endif // __MODEL_H__