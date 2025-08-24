#include "Model.h"
#include <iostream>
#include "LossMeanSquaredError.h"
#include "LossCategoricalCrossEntropy.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;


void NEURAL_NETWORK::Model::add(std::unique_ptr<LayerBase> layer)
{
	layers_.push_back(std::move(layer));
}

void NEURAL_NETWORK::Model::set(std::unique_ptr<Loss> loss, 
								std::unique_ptr<Optimizer> optimizer, 
								std::unique_ptr<Accuracy> accuracy)
{
	loss_ = std::move(loss);
	optimizer_ = std::move(optimizer);
	accuracy_ = std::move(accuracy);
}

void NEURAL_NETWORK::Model::finalize()
{
	input_layer_ = std::make_unique<LayerInput>();

	int layer_count = layers_.size();
	
	trainable_layers_.clear();

	for (int i = 0; i < layer_count; ++i)
	{
		if (i == 0)
		{
			layers_[i]->setPrev(input_layer_.get());
			if (layer_count > 1) 
			{
				layers_[i]->setNext(layers_[i+1].get());
			} 
			else 
			{
				layers_[i]->setNext(loss_.get());
			}
		}
		else if (i < layer_count - 1)
		{
			layers_[i]->setPrev(layers_[i-1].get());
			layers_[i]->setNext(layers_[i+1].get());
		}
		else
		{
			layers_[i]->setPrev(layers_[i-1].get());
			layers_[i]->setNext(loss_.get());
		}
		
		if (LayerDense* dense_layer = dynamic_cast<LayerDense*>(layers_[i].get())) 
		{
			trainable_layers_.push_back(dense_layer);
		}
	}

	loss_->RememberTrainableLayers(trainable_layers_);

	if (dynamic_cast<ActivationSoftmax*>(layers_.back().get()))
	{
		if (dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get()))
		{
			softmax_loss_ = std::make_unique<ActivationSoftmaxLossCategoricalCrossEntropy>();
			softmax_classifier = true;
		}
	}
	else if (dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get()))
	{
		if (dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get()))
		{
			softmax_classifier = true;
		}
	}
}

void NEURAL_NETWORK::Model::train(const Eigen::MatrixXd& X, Eigen::MatrixXd& y, 
								  int epochs, int print_every, 
								  const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& validation_data,
								  bool plot_validation_data)
{
	accuracy_->init(y);

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		forward(X, true);
		double reg_loss = 0.0;
		if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
		{
			mse->forwardDouble(output_, y);
			reg_loss = loss_->RegularizationLoss();
		} 
		else 
		{
			loss_->CalculateLoss(output_, y.cast<int>(), true);
			reg_loss = loss_->GetRegularizationLoss();
			
			if (softmax_classifier) 
			{
				if (auto* combined = dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get())) 
				{
					combined->storeTargets(y.cast<int>());
				}
			}
		}
		double loss = loss_->GetLoss() + reg_loss;
		Eigen::MatrixXd predictions = layers_.back()->predictions();
		double accuracy = accuracy_->Calculate(predictions, y);

		backward(output_, y);

		optimizer_->PreUpdateParameters();
		for (LayerDense* layer : trainable_layers_)
		{
			optimizer_->UpdateParameters(*layer);
		}
		optimizer_->PostUpdateParameters();
		if (!(epoch % print_every))
		{
			std::cout << "Epoch: " << epoch 
					  << ", Accuracy: " << accuracy 
					  << ", Loss: " << loss 
					  << " (Data loss: " << loss_->GetLoss()
					  << " | Regularization loss: " << reg_loss
					  << "), Learning Rate: " << optimizer_->GetLearningRate()
					  << '\n';
		}
	}

	if (validation_data.first.size() > 0 && validation_data.second.size() > 0)
	{
		Eigen::MatrixXd X_validation = validation_data.first;
		Eigen::MatrixXd y_validation = validation_data.second;

		forward(X_validation, false);
		double val_reg_loss = 0.0;
		if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
		{
			mse->forwardDouble(output_, y_validation);
			val_reg_loss = loss_->RegularizationLoss();
		} 
		else 
		{
			loss_->CalculateLoss(output_, y_validation.cast<int>());
			val_reg_loss = loss_->GetRegularizationLoss();
			
			if (softmax_classifier) 
			{
				if (auto* combined = dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get())) 
				{
					combined->storeTargets(y_validation.cast<int>());
				}
			}
		}
		double val_loss = loss_->GetLoss() + val_reg_loss;
		Eigen::MatrixXd val_predictions = layers_.back()->predictions();
		double val_accuracy = accuracy_->Calculate(val_predictions, y_validation);
		std::cout << "Validation Accuracy: " << val_accuracy 
				  << ", Validation Loss: " << val_loss << '\n';
		if (plot_validation_data)
		{
			std::vector<double> x_test_vec(X_validation.data(), X_validation.data() + X_validation.size());
			std::vector<double> y_test_vec(y_validation.data(), y_validation.data() + y_validation.size());		
			std::vector<double> y_pred_vec(val_predictions.data(), val_predictions.data() + val_predictions.size());		
			plt::figure_size(1000, 600);
			plt::named_plot("True", x_test_vec, y_test_vec, "b-");
			plt::named_plot("Predicted", x_test_vec, y_pred_vec, "r-");
			plt::title("Sine Wave Regression");
			plt::xlabel("X");
			plt::ylabel("y");
			plt::legend();
			plt::show();
		}
	}
}

void NEURAL_NETWORK::Model::forward(const Eigen::MatrixXd& inputs, bool training)
{
	input_layer_->forward(inputs, training);
	
	for (const auto& layer : layers_)
	{
		layer->forward(static_cast<LayerBase*>(layer->getPrev())->GetOutput(), training);
	}

	output_ = layers_.back()->GetOutput();
}

void NEURAL_NETWORK::Model::backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& targets)
{
	auto start_iter = layers_.rbegin();
	if (softmax_classifier)
	{
		if (auto* combined = dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get()))
		{
			combined->backward(output);
		}
		else if (softmax_loss_)
		{
			softmax_loss_->storeTargets(targets.cast<int>());
			softmax_loss_->backward(output);
			layers_.back()->SetDInput(softmax_loss_->GetDInput());
		}
		start_iter++;
	}
	else
	{
		if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get()))
		{
			mse->backwardDouble(output, targets);
			layers_.back()->backward(mse->GetDInput());
		} 
		else
		{
			loss_->backward(output, targets.cast<int>());
			layers_.back()->backward(loss_->GetDInput());
		}
		start_iter++;
	}
	
	for (auto layer = start_iter; layer != layers_.rend(); ++layer)
	{
		(*layer)->backward(static_cast<LayerBase*>((*layer)->getNext())->GetDInput());
	}
}
