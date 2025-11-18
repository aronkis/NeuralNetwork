#include "Model.h"
#include "LayerDense.h"
#include "BatchNormalization.h"
#include "Helpers.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

struct ModelConfig 
{
    std::vector<std::string> layer_types;
    std::vector<std::vector<double>> layer_params;
    std::string loss_type;
    std::vector<double> loss_params;
    std::string optimizer_type;
    std::vector<double> optimizer_params;
    std::string accuracy_type;
    std::vector<double> accuracy_params;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> parameters;
    bool softmax_classifier_;
};

void NEURAL_NETWORK::Model::Add(std::shared_ptr<LayerBase> layer)
{
	layers_.push_back(std::move(layer));
}

void NEURAL_NETWORK::Model::Set(std::unique_ptr<Loss> loss, 
								std::unique_ptr<Accuracy> accuracy,
								std::unique_ptr<Optimizer> optimizer)
{
	if (loss)
	{
		loss_ = std::move(loss);
	}
	if (optimizer)
	{
		optimizer_ = std::move(optimizer);
	}
	if (accuracy)
	{
		accuracy_ = std::move(accuracy);
	}
}

void NEURAL_NETWORK::Model::Finalize()
{
	input_layer_ = std::make_shared<LayerInput>();

	int layer_count = layers_.size();
	
	trainable_layers_.clear();

	for (int i = 0; i < layer_count; i++)
	{
		if (i == 0)
		{
			layers_[i]->setPrev(input_layer_);
			if (layer_count > 1)
			{
				layers_[i]->setNext(layers_[i+1]);
			}
			else
			{
				layers_[i]->setNext(std::shared_ptr<LayerBase>());
			}
		}
		else if (i < layer_count - 1)
		{
			layers_[i]->setPrev(layers_[i-1]);
			layers_[i]->setNext(layers_[i+1]);
		}
		else
		{
			layers_[i]->setPrev(layers_[i-1]);
			layers_[i]->setNext(std::shared_ptr<LayerBase>());
		}
		
		if (auto layer = std::dynamic_pointer_cast<LayerDense>(layers_[i]))
		{
			trainable_layers_.emplace_back(layer);
		}
		else if (auto layer = std::dynamic_pointer_cast<Convolution2D>(layers_[i]))
		{
			trainable_layers_.emplace_back(layer);
		}
		else if (auto layer = std::dynamic_pointer_cast<BatchNormalization>(layers_[i]))
		{
			trainable_layers_.emplace_back(layer);
		}
	}

	if (loss_)
	{
		std::vector<std::weak_ptr<LayerBase>> weak_trainables;

		for (auto &trainable_layer_ : trainable_layers_)
		{
			weak_trainables.emplace_back(trainable_layer_);
		}
		loss_->RememberTrainableLayers(weak_trainables);
	}
	
	if (dynamic_cast<ActivationSoftmax*>(layers_.back().get()))
	{
		if (dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get()))
		{
			softmax_loss_ = std::make_unique<ActivationSoftmaxLossCategoricalCrossEntropy>();
			softmax_classifier_ = true;
		}
	}
	else if (dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get()))
	{
		if (dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get()))
		{
			softmax_classifier_ = true;
		}
	}
}

void NEURAL_NETWORK::Model::Evaluate(const Eigen::MatrixXd& X, 
                                     const Eigen::MatrixXd& y, 
                                     int batch_size)
{
    if (!loss_ || !accuracy_)
    {
        std::cerr << "Model::Evaluate error: model not configured. "
                  << "Call Set(loss, optimizer, accuracy) before Evaluate().\n";
        return;
    }

	int validation_steps = 1;
	Eigen::MatrixXd all_val_predictions; 
	if (batch_size > 1)
	{
		if (X.size() > 0 && y.size() > 0)
		{
			validation_steps = X.rows() / batch_size;
			if (validation_steps * batch_size < X.rows())
			{
				validation_steps++;
			}
		}
	}

	loss_->NewPass();
	accuracy_->NewPass();

	int output_cols = layers_.back()->GetOutput().cols();
	if (output_cols == 0 && y.cols() > 0)
	{
		output_cols = y.cols();
	}
	all_val_predictions.resize(X.rows(), output_cols);

	for (int validation_step = 0; 
		 validation_step < validation_steps; 
		 validation_step++)
	{
		int start_idx = validation_step * batch_size;
		int end_idx;
		if (batch_size > 1)
		{
			end_idx = std::min((validation_step + 1) * batch_size, 
							   static_cast<int>(X.rows()));
		}
		else
		{
			end_idx = X.rows();
		}
		const Eigen::Block<const Eigen::MatrixXd> batch_X = X.block(start_idx, 
																	0, 
																	end_idx - start_idx, 
																	X.cols());
		const Eigen::Block<const Eigen::MatrixXd> batch_y = y.block(start_idx, 
																	0, 
																	end_idx - start_idx, 
																	y.cols());

		forward(batch_X, false);
		
		if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
		{
			mse->forward(output_, batch_y);
			loss_->CalculateLoss(output_, batch_y, false);
		} 
		else 
		{
			loss_->CalculateLoss(output_, batch_y);

			if (softmax_classifier_) 
			{
				if (auto* combined = dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get())) 
				{
					combined->storeTargets(batch_y.cast<int>());
				}
			}
		}
		Eigen::MatrixXd val_predictions = layers_.back()->predictions();
		all_val_predictions.block(start_idx, 
								  0, 
								  val_predictions.rows(), 
								  val_predictions.cols()) = val_predictions;
		accuracy_->Calculate(val_predictions, batch_y);
	}

	loss_->CalculateAccumulatedLoss();
	double val_loss = loss_->GetAccumulatedLoss();
	accuracy_->CalculateAccumulated();
	double val_accuracy = accuracy_->GetAccumulatedAccuracy();
	std::cout << "Validation Accuracy: " << val_accuracy
			<< ", Validation Loss: " << val_loss << '\n';
}

void NEURAL_NETWORK::Model::Train(const Eigen::MatrixXd& X, 
								  const Eigen::MatrixXd& y,
				   				  int batch_size, int epochs, int print_every, 
				   				  const Eigen::MatrixXd& X_val, 
								  const Eigen::MatrixXd& y_val)
{
	if (!optimizer_)
	{
		std::cerr << "Model::Train error: model not configured. "
				  << "Call Set(loss, optimizer, accuracy) before Train(). "
				  << "Evaluation models cannot use Model::Train().\n";
		return;
	}

	if (X.rows() == 0 || X.cols() == 0 || y.rows() == 0 || y.cols() == 0)
	{
		std::cout << "Model::Train notice: received empty training data; skipping optimization." << std::endl;
		if (X_val.size() > 0 && y_val.size() > 0)
		{
			int effective_batch = std::max(batch_size, 1);
			Evaluate(X_val, y_val, effective_batch);
		}
		return;
	}

	accuracy_->init(y);

	int train_steps = 1;

	if (batch_size > 1)
	{
		train_steps = X.rows() / batch_size;
		if (train_steps * batch_size < X.rows())
		{
			train_steps++;
		}
	}

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "\n========== Epoch " << (epoch + 1) << "/" << epochs << " ==========\n";
		
		// Reset loss and accuracy accumulators for new epoch
		loss_->NewPass();
		accuracy_->NewPass();
		
		for (int step = 0; step < train_steps; step++)
		{
			int start_idx = step * batch_size;
			int end_idx;
			if (batch_size > 1)
			{
				end_idx = std::min((step + 1) * batch_size, 
									static_cast<int>(X.rows()));
			}
			else
			{
				end_idx = X.rows();
			}
			const Eigen::Block<const Eigen::MatrixXd> batch_X = X.block(start_idx, 
																		0, 
																		end_idx - start_idx, 
																		X.cols());
			const Eigen::Block<const Eigen::MatrixXd> batch_y = y.block(start_idx, 
																		0, 
																		end_idx - start_idx, 
																		y.cols());

			forward(batch_X, true);

			double reg_loss = 0.0;
			if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
			{
				mse->forward(output_, batch_y);
				loss_->CalculateLoss(output_, batch_y, true);
				loss_->RegularizationLoss();
				reg_loss = loss_->GetRegularizationLoss();
			} 
			else 
			{
				loss_->CalculateLoss(output_, batch_y, true);
				loss_->GetRegularizationLoss();
				reg_loss = loss_->GetRegularizationLoss();

				if (softmax_classifier_) 
				{
					if (auto* combined = dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layers_.back().get())) 
					{
						combined->storeTargets(batch_y.cast<int>());
					}
				}
			}
			double loss = loss_->GetLoss() + reg_loss;
			Eigen::MatrixXd predictions = layers_.back()->predictions();
			accuracy_->Calculate(predictions, batch_y);
			double accuracy = accuracy_->GetAccuracy();

			backward(output_, batch_y);

			optimizer_->PreUpdateParameters();

			for (auto &layer_sp : trainable_layers_)
			{
				if (layer_sp)
				{
					if (auto dense_layer = std::dynamic_pointer_cast<LayerDense>(layer_sp))
					{
						optimizer_->UpdateParameters(*dense_layer);
					}
					else if (auto conv_layer = std::dynamic_pointer_cast<Convolution2D>(layer_sp))
					{
						optimizer_->UpdateParameters(*conv_layer);
					}
					else if (auto batchnorm_layer = std::dynamic_pointer_cast<BatchNormalization>(layer_sp))
					{
						optimizer_->UpdateParameters(*batchnorm_layer);
					}
				}
			}

			optimizer_->PostUpdateParameters();
			if (batch_size > 1 && 
				(!(step % print_every) || 
				 (step == train_steps - 1)))
			{
				double progress = 100.0 * (step + 1) / train_steps;
                double learning_rate = optimizer_->GetLearningRate();

				std::cout << "Step: " << step 
						  << " [" << std::fixed << std::setprecision(2) << progress << "%]"
						  << std::defaultfloat << std::setprecision(6)
						  << ", Accuracy: " << accuracy
						  << ", Loss: " << loss 
						  << ", LR: " << learning_rate << std::endl;
			}
		}
		loss_->CalculateAccumulatedLoss(true);
		loss_->GetRegularizationLoss();
		double epoch_reg_loss = loss_->GetRegularizationLoss();
		double epoch_loss = loss_->GetAccumulatedLoss() + epoch_reg_loss;

		accuracy_->CalculateAccumulated();
		double epoch_acc = accuracy_->GetAccumulatedAccuracy();

		std::cout << "Training: "
				<< "Accuracy: " << epoch_acc
				<< ", Loss: " << epoch_loss
				<< " (Data loss: " << loss_->GetAccumulatedLoss()
				<< " | Regularization loss: " << epoch_reg_loss
				<< "), Learning Rate: " << optimizer_->GetLearningRate()
				<< '\n';
	}

	if (X_val.size() > 0 && y_val.size() > 0)
	{
		Evaluate(X_val, y_val, batch_size);
	}
}

Eigen::MatrixXd NEURAL_NETWORK::Model::Predict(const Eigen::MatrixXd& X, 
											   int batch_size)
{
	int prediction_steps = 1;

	if (batch_size > 1)
	{
		prediction_steps = X.rows() / batch_size;
		if (prediction_steps * batch_size < X.rows())
		{
			prediction_steps++;
		}
	}
	long output_size = layers_.back()->predictions().cols();
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(X.rows(), output_size);

	for (int prediction_step = 0; 
		 prediction_step < prediction_steps; 
		 prediction_step++)
	{
		int start_idx = prediction_step * batch_size;
		int end_idx;
		
		if (batch_size > 1)
		{
			end_idx = std::min((prediction_step + 1) * batch_size, 
								static_cast<int>(X.rows()));
		}
		else
		{
			end_idx = X.rows();
		}

		const Eigen::Block<const Eigen::MatrixXd> batch_X = X.block(start_idx, 
																	0, 
																	end_idx - start_idx, 
																	X.cols());

		forward(batch_X, false);

		output.block(start_idx, 
		 			  0, 
		 			  batch_X.rows(), 
		 			  output_size) = layers_.back()->predictions();
	}

	return output;
}

std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> NEURAL_NETWORK::Model::GetParameters() const
{
	std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> params;
	
	for (const auto& layer : trainable_layers_)
	{
		params.push_back(layer->GetParameters());
	}

	return params;
}

void NEURAL_NETWORK::Model::SetParameters(const std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>>& params)
{
	for (size_t i = 0; i < trainable_layers_.size(); i++)
	{
		trainable_layers_[i]->SetParameters(params[i].first, params[i].second);
	}
}

void NEURAL_NETWORK::Model::SaveParameters(const std::string& path) const
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) 
	{
		std::cerr << "Model::SaveParameters error: Failed to open file for writing"
				  << std::endl;
		return;
	}

    auto params = GetParameters();

    size_t numParams = params.size();
    NEURAL_NETWORK::Serialization::WriteRaw(ofs, &numParams, sizeof(numParams));
    for (const auto& param : params) 
	{
        NEURAL_NETWORK::Serialization::WriteMatrix(ofs, param.first);
        NEURAL_NETWORK::Serialization::WriteRowVector(ofs, param.second);
    }

    ofs.close();
}

void NEURAL_NETWORK::Model::LoadParameters(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
	{
		std::cerr << "Model::LoadParameters error: Failed to open file for reading"
				  << std::endl;
		return;
	}

    size_t numParams;
    NEURAL_NETWORK::Serialization::ReadRaw(ifs, &numParams, sizeof(numParams));
    std::vector<std::pair<Eigen::MatrixXd, Eigen::RowVectorXd>> params(numParams);

    for (auto& param : params) 
	{
        param.first = NEURAL_NETWORK::Serialization::ReadMatrix(ifs);
        param.second = NEURAL_NETWORK::Serialization::ReadRowVector(ifs);
    }

    SetParameters(params);
    ifs.close();
}

void NEURAL_NETWORK::Model::SaveModel(const std::string& path) const
{
	ModelConfig config;
    config.softmax_classifier_ = softmax_classifier_;
    config.parameters = GetParameters();

    for (const auto& layer : layers_) 
	{
        if (auto* dense = dynamic_cast<LayerDense*>(layer.get())) 
		{
            config.layer_types.push_back("LayerDense");
            std::vector<double> params = {
                static_cast<double>(dense->GetWeights().cols()),
                static_cast<double>(dense->GetWeights().rows()),
                dense->GetWeightRegularizerL1(),
                dense->GetWeightRegularizerL2(),
                dense->GetBiasRegularizerL1(),
                dense->GetBiasRegularizerL2()
            };
            config.layer_params.push_back(params);
        } 
		else if (auto* conv = dynamic_cast<Convolution2D*>(layer.get()))
		{
			config.layer_types.push_back("Convolution2D");
			std::vector<double> params = {
				static_cast<double>(conv->GetNumberOfFilters()),
				static_cast<double>(conv->GetFilterHeight()),
				static_cast<double>(conv->GetFilterWidth()),
				static_cast<double>(conv->GetInputHeight()),
				static_cast<double>(conv->GetInputWidth()),
				static_cast<double>(conv->GetInputChannels()),
				static_cast<double>(conv->GetPadding()),
				static_cast<double>(conv->GetStrideHeight()),
				static_cast<double>(conv->GetStrideWidth()),
				conv->GetWeightRegularizerL1(),
				conv->GetWeightRegularizerL2(),
				conv->GetBiasRegularizerL1(),
				conv->GetBiasRegularizerL2()
			};
			config.layer_params.push_back(params);
		} 
		else if (auto* conv1d = dynamic_cast<Convolution1D*>(layer.get()))
		{
			config.layer_types.push_back("Convolution1D");
			std::vector<double> params = {
				static_cast<double>(conv1d->GetNumberOfFilters()),
				static_cast<double>(conv1d->GetFilterLength()),
				static_cast<double>(conv1d->GetInputLength()),
				static_cast<double>(conv1d->GetInputChannels()),
				static_cast<double>(conv1d->GetPadding()),
				static_cast<double>(conv1d->GetStride()),
				conv1d->GetWeightRegularizerL1(),
				conv1d->GetWeightRegularizerL2(),
				conv1d->GetBiasRegularizerL1(),
				conv1d->GetBiasRegularizerL2()
			};
			config.layer_params.push_back(params);
		}
		else if (auto* dropout = dynamic_cast<LayerDropout*>(layer.get())) 
		{
            config.layer_types.push_back("LayerDropout");
            config.layer_params.push_back({dropout->GetRate()});
        } 
		else if (auto* maxpool = dynamic_cast<MaxPooling*>(layer.get()))
		{
            config.layer_types.push_back("MaxPooling");
            std::vector<double> params = {
                static_cast<double>(maxpool->GetPoolSize()),
                static_cast<double>(maxpool->GetStride()),
                static_cast<double>(maxpool->GetInputHeight()),
                static_cast<double>(maxpool->GetInputWidth()),
                static_cast<double>(maxpool->GetInputChannels())
            };
            config.layer_params.push_back(params);
        }
		else if (auto* maxpool1d = dynamic_cast<MaxPooling1D*>(layer.get()))
		{
			config.layer_types.push_back("MaxPooling1D");
			std::vector<double> params = {
				static_cast<double>(maxpool1d->GetPoolSize()),
				static_cast<double>(maxpool1d->GetStride()),
				static_cast<double>(maxpool1d->GetInputLength()),
				static_cast<double>(maxpool1d->GetInputChannels())
			};
			config.layer_params.push_back(params);
		}
		else if (auto* batchnorm = dynamic_cast<BatchNormalization*>(layer.get()))
		{
            config.layer_types.push_back("BatchNormalization");
            std::vector<double> params = 
			{
                static_cast<double>(batchnorm->GetNumFeatures()),

            };
            config.layer_params.push_back(params);
        } 
		else if (dynamic_cast<LayerInput*>(layer.get())) 
		{
            config.layer_types.push_back("LayerInput");
            config.layer_params.push_back({});
        } 
		else if (dynamic_cast<ActivationReLU*>(layer.get())) 
		{
            config.layer_types.push_back("ActivationReLU");
            config.layer_params.push_back({});
        } 
		else if (dynamic_cast<ActivationSigmoid*>(layer.get())) 
		{
            config.layer_types.push_back("ActivationSigmoid");
            config.layer_params.push_back({});
        } 
		else if (dynamic_cast<ActivationSoftmax*>(layer.get())) 
		{
            config.layer_types.push_back("ActivationSoftmax");
            config.layer_params.push_back({});
        } 
		else if (dynamic_cast<ActivationSoftmaxLossCategoricalCrossEntropy*>(layer.get())) 
		{
            config.layer_types.push_back("ActivationSoftmaxLossCategoricalCrossEntropy");
            config.layer_params.push_back({});
        } 
		else if (dynamic_cast<ActivationLinear*>(layer.get())) 
		{
            config.layer_types.push_back("ActivationLinear");
            config.layer_params.push_back({});
        } 
		else 
		{
            std::cerr << "Unknown layer type in SaveModel\n";
        }
    }

    if (dynamic_cast<LossBinaryCrossEntropy*>(loss_.get())) 
	{
        config.loss_type = "LossBinaryCrossEntropy";
        config.loss_params = {};
    } 
	else if (dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get())) 
	{
        config.loss_type = "LossCategoricalCrossEntropy";
        config.loss_params = {};
    } 
	else if (dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
	{
        config.loss_type = "LossMeanSquaredError";
        config.loss_params = {};
    } 
	else if (dynamic_cast<LossMeanAbsoluteError*>(loss_.get())) 
	{
        config.loss_type = "LossMeanAbsoluteError";
        config.loss_params = {};
    } 
	else 
	{
        config.loss_type = "";
    }

    if (auto* adam = dynamic_cast<Adam*>(optimizer_.get())) 
	{
        config.optimizer_type = "Adam";
        config.optimizer_params = {
            adam->GetLearningRate(),
            adam->GetDecay(),
            adam->GetBeta1(),
            adam->GetBeta2(),
            adam->GetEpsilon()
        };
    } 
	else if (auto* sgd = dynamic_cast<StochasticGradientDescent*>(optimizer_.get())) 
	{
        config.optimizer_type = "StochasticGradientDescent";
        config.optimizer_params = {
            sgd->GetLearningRate(),
            sgd->GetDecay(),
            sgd->GetMomentum()
        };
    } 
	else if (auto* adagrad = dynamic_cast<AdaGrad*>(optimizer_.get())) 
	{
        config.optimizer_type = "AdaGrad";
        config.optimizer_params = {
            adagrad->GetLearningRate(),
            adagrad->GetDecay(),
            adagrad->GetEpsilon()
        };
    } 
	else if (auto* rmsprop = dynamic_cast<RMSProp*>(optimizer_.get())) 
	{
        config.optimizer_type = "RMSProp";
        config.optimizer_params = {
            rmsprop->GetLearningRate(),
            rmsprop->GetDecay(),
            rmsprop->GetRho(),
            rmsprop->GetEpsilon()
        };
    } 
	else 
	{
        config.optimizer_type = "";
    }

    if (dynamic_cast<AccuracyCategorical*>(accuracy_.get())) 
	{
        config.accuracy_type = "AccuracyCategorical";
        config.accuracy_params = {};
    } 
	else if (auto* reg = dynamic_cast<AccuracyRegression*>(accuracy_.get())) 
	{
        config.accuracy_type = "AccuracyRegression";
        config.accuracy_params = {reg->GetEpsilon()};
    } 
	else 
	{
        config.accuracy_type = "";
    }

	std::filesystem::path target_path = std::filesystem::path(path).lexically_normal();
	target_path.make_preferred();

	const std::filesystem::path parent = target_path.parent_path();
	if (!parent.empty())
	{
		std::error_code ec;
		std::filesystem::create_directories(parent, ec);
		if (ec)
		{
			std::cerr << "Model::SaveModel error: failed to create directories for "
					  << parent << ": " << ec.message() << '\n';
			return;
		}
	}

	std::ofstream ofs(target_path, std::ios::binary);
	if (!ofs)
	{
		std::cerr << "Model::SaveModel error: failed to open file for writing at "
				  << target_path << '\n';
		return;
	}

    size_t numLayers = config.layer_types.size();
    NEURAL_NETWORK::Serialization::WriteRaw(ofs, &numLayers, sizeof(numLayers));
    
	for (size_t i = 0; i < numLayers; i++) 
	{
        NEURAL_NETWORK::Serialization::WriteString(ofs, config.layer_types[i]);
        NEURAL_NETWORK::Serialization::WriteVectorDouble(ofs, config.layer_params[i]);
    }

    NEURAL_NETWORK::Serialization::WriteString(ofs, config.loss_type);
    NEURAL_NETWORK::Serialization::WriteVectorDouble(ofs, config.loss_params);
    NEURAL_NETWORK::Serialization::WriteString(ofs, config.optimizer_type);
    NEURAL_NETWORK::Serialization::WriteVectorDouble(ofs, config.optimizer_params);
    NEURAL_NETWORK::Serialization::WriteString(ofs, config.accuracy_type);
    NEURAL_NETWORK::Serialization::WriteVectorDouble(ofs, config.accuracy_params);

    size_t numParams = config.parameters.size();
    NEURAL_NETWORK::Serialization::WriteRaw(ofs, &numParams, sizeof(numParams));

    for (const auto& param : config.parameters) 
	{
        NEURAL_NETWORK::Serialization::WriteMatrix(ofs, param.first);
        NEURAL_NETWORK::Serialization::WriteRowVector(ofs, param.second);
    }

    NEURAL_NETWORK::Serialization::WriteRaw(ofs, 
											&config.softmax_classifier_, 
											sizeof(config.softmax_classifier_));

	ofs.close();
	std::cout << "Model saved to " << target_path << std::endl;
}

void NEURAL_NETWORK::Model::LoadModel(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
	{
		std::cerr << "Failed to open file for reading.\n";
		return;
	}

    ModelConfig config;

    size_t numLayers;
    NEURAL_NETWORK::Serialization::ReadRaw(ifs, &numLayers, sizeof(numLayers));
    
	config.layer_types.resize(numLayers);
    config.layer_params.resize(numLayers);

    for (size_t i = 0; i < numLayers; i++) 
	{
        config.layer_types[i] = NEURAL_NETWORK::Serialization::ReadString(ifs);
        config.layer_params[i] = NEURAL_NETWORK::Serialization::ReadVectorDouble(ifs);
    }

    config.loss_type = NEURAL_NETWORK::Serialization::ReadString(ifs);
    config.loss_params = NEURAL_NETWORK::Serialization::ReadVectorDouble(ifs);
    config.optimizer_type = NEURAL_NETWORK::Serialization::ReadString(ifs);
    config.optimizer_params = NEURAL_NETWORK::Serialization::ReadVectorDouble(ifs);
    config.accuracy_type = NEURAL_NETWORK::Serialization::ReadString(ifs);
    config.accuracy_params = NEURAL_NETWORK::Serialization::ReadVectorDouble(ifs);

    size_t numParams;
    NEURAL_NETWORK::Serialization::ReadRaw(ifs, &numParams, sizeof(numParams));
    
	config.parameters.resize(numParams);

    for (auto& param : config.parameters) 
	{
        param.first = NEURAL_NETWORK::Serialization::ReadMatrix(ifs);
        param.second = NEURAL_NETWORK::Serialization::ReadRowVector(ifs);
    }

    NEURAL_NETWORK::Serialization::ReadRaw(ifs, 
										   &config.softmax_classifier_, 
										   sizeof(config.softmax_classifier_));

	ifs.close();
	
	layers_.clear();
	trainable_layers_.clear();
	loss_.reset();
	optimizer_.reset();
	accuracy_.reset();
	softmax_classifier_ = false;

	if (config.layer_types.empty())
	{
		std::cerr << "Model::LoadModel warning: no layers found in saved model; keeping model empty." << std::endl;
		return;
	}

	softmax_classifier_ = config.softmax_classifier_;

    for (size_t i = 0; i < config.layer_types.size(); i++) 
	{
        const auto& type = config.layer_types[i];
        const auto& params = config.layer_params[i];

        if (type == "LayerInput") 
		{
            Add(std::make_shared<LayerInput>());
        } 
		else if (type == "LayerDense") 
		{
            if (params.size() < 6)
			{
				std::cerr << "Model::LoadModel warning: LayerDense entry missing parameters; aborting load." << std::endl;
				return;
			}
            int n_inputs = static_cast<int>(params[0]);
            int n_neurons = static_cast<int>(params[1]);
            double wl1 = params[2];
            double wl2 = params[3];
            double bl1 = params[4];
            double bl2 = params[5];
            Add(std::make_shared<LayerDense>(n_inputs, n_neurons, 
											 wl1, wl2, 
											 bl1, bl2));
        }
		else if (type == "LayerDropout")
		{
            if (params.size() < 1)
			{
				std::cerr << "Model::LoadModel warning: LayerDropout entry missing rate parameter; aborting load." << std::endl;
				return;
			}
            double rate = params[0];
            Add(std::make_shared<LayerDropout>(rate));
        }
		else if (type == "MaxPooling")
		{
            if (params.size() < 5)
			{
				std::cerr << "Model::LoadModel warning: MaxPooling entry missing parameters; aborting load." << std::endl;
				return;
			}
            int pool_size = static_cast<int>(params[0]);
            int stride = static_cast<int>(params[1]);
            int input_height = static_cast<int>(params[2]);
            int input_width = static_cast<int>(params[3]);
            int input_channels = static_cast<int>(params[4]);
            // Use batch_size = 1 as default, it will resize automatically in forward()
            int batch_size = 1;
            Add(std::make_shared<MaxPooling>(batch_size, pool_size, input_height,
                                           input_width, input_channels, stride));
        }
		else if (type == "MaxPooling1D")
		{
			if (params.size() < 4)
			{
				std::cerr << "Model::LoadModel warning: MaxPooling1D entry missing parameters; aborting load." << std::endl;
				return;
			}
			int pool_size = static_cast<int>(params[0]);
			int stride = static_cast<int>(params[1]);
			int input_length = static_cast<int>(params[2]);
			int input_channels = static_cast<int>(params[3]);
			int batch_size = 1;
			Add(std::make_shared<MaxPooling1D>(batch_size, pool_size, input_length,
												input_channels, stride));
		}
		else if (type == "BatchNormalization")
		{
            if (params.size() < 1)
			{
				std::cerr << "Model::LoadModel warning: BatchNormalization entry missing parameters; aborting load." << std::endl;
				return;
			}
            int num_features = static_cast<int>(params[0]);
            Add(std::make_shared<BatchNormalization>(num_features));
        }
		else if (type == "Convolution2D")
		{
			if (params.size() < 9)
			{
				std::cerr << "Model::LoadModel warning: Convolution2D entry missing parameters; aborting load." << std::endl;
				return;
			}
			int number_of_filters = static_cast<int>(params[0]);
			int filter_height = static_cast<int>(params[1]);
			int filter_width = static_cast<int>(params[2]);
			int input_height = static_cast<int>(params[3]);
			int input_width = static_cast<int>(params[4]);
			int input_channels = static_cast<int>(params[5]);
			int padding = static_cast<int>(params[6]);
			int stride_height = static_cast<int>(params[7]);
			int stride_width = static_cast<int>(params[8]);

			double weight_l1 = (params.size() > 9) ? params[9] : 0.0;
			double weight_l2 = (params.size() > 10) ? params[10] : 0.0;
			double bias_l1 = (params.size() > 11) ? params[11] : 0.0;
			double bias_l2 = (params.size() > 12) ? params[12] : 0.0;

			Add(std::make_shared<Convolution2D>(number_of_filters, filter_height, filter_width,
											  input_height, input_width, input_channels,
											  padding, stride_height, stride_width,
											  weight_l1, weight_l2, bias_l1, bias_l2));
		}
		else if (type == "Convolution1D")
		{
			if (params.size() < 6)
			{
				std::cerr << "Model::LoadModel warning: Convolution1D entry missing parameters; aborting load." << std::endl;
				return;
			}
			int number_of_filters = static_cast<int>(params[0]);
			int filter_length = static_cast<int>(params[1]);
			int input_length = static_cast<int>(params[2]);
			int input_channels = static_cast<int>(params[3]);
			int padding = static_cast<int>(params[4]);
			int stride = static_cast<int>(params[5]);

			double weight_l1 = (params.size() > 6) ? params[6] : 0.0;
			double weight_l2 = (params.size() > 7) ? params[7] : 0.0;
			double bias_l1 = (params.size() > 8) ? params[8] : 0.0;
			double bias_l2 = (params.size() > 9) ? params[9] : 0.0;

			Add(std::make_shared<Convolution1D>(number_of_filters, filter_length,
											 input_length, input_channels,
											 padding, stride,
											 weight_l1, weight_l2,
											 bias_l1, bias_l2));
		}
		else if (type == "ActivationReLU") 
		{
            Add(std::make_shared<ActivationReLU>());
        } 
		else if (type == "ActivationSigmoid") 
		{
            Add(std::make_shared<ActivationSigmoid>());
        } 
		else if (type == "ActivationSoftmax") 
		{
            Add(std::make_shared<ActivationSoftmax>());
        } 
		else if (type == "ActivationSoftmaxLossCategoricalCrossEntropy") 
		{
            Add(std::make_shared<ActivationSoftmaxLossCategoricalCrossEntropy>());
        } 
		else if (type == "ActivationLinear") 
		{
            Add(std::make_shared<ActivationLinear>());
        }
    }

    if (config.loss_type == "LossBinaryCrossEntropy") 
	{
        loss_ = std::make_unique<LossBinaryCrossEntropy>();
    } 
	else if (config.loss_type == "LossCategoricalCrossEntropy") 
	{
        loss_ = std::make_unique<LossCategoricalCrossEntropy>();
    } 
	else if (config.loss_type == "LossMeanSquaredError") 
	{
        loss_ = std::make_unique<LossMeanSquaredError>();
    } 
	else if (config.loss_type == "LossMeanAbsoluteError") 
	{
        loss_ = std::make_unique<LossMeanAbsoluteError>();
    }

	if (config.optimizer_type == "Adam") 
	{
		if (config.optimizer_params.size() < 5)
		{
			std::cerr << "Model::LoadModel warning: Adam optimizer entry missing parameters; aborting load." << std::endl;
			return;
		}
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double b1 = config.optimizer_params[2];
        double b2 = config.optimizer_params[3];
        double eps = config.optimizer_params[4];
        optimizer_ = std::make_unique<Adam>(lr, decay, b1, b2, eps);
    } 
	else if (config.optimizer_type == "StochasticGradientDescent") 
	{
		if (config.optimizer_params.size() < 3)
		{
			std::cerr << "Model::LoadModel warning: SGD optimizer entry missing parameters; aborting load." << std::endl;
			return;
		}
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double momentum = config.optimizer_params[2];
        optimizer_ = std::make_unique<StochasticGradientDescent>(lr, decay, momentum);
    } 
	else if (config.optimizer_type == "AdaGrad") 
	{
		if (config.optimizer_params.size() < 3)
		{
			std::cerr << "Model::LoadModel warning: AdaGrad optimizer entry missing parameters; aborting load." << std::endl;
			return;
		}
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double eps = config.optimizer_params[2];
        optimizer_ = std::make_unique<AdaGrad>(lr, decay, eps);
    } 
	else if (config.optimizer_type == "RMSProp") 
	{
		if (config.optimizer_params.size() < 4)
		{
			std::cerr << "Model::LoadModel warning: RMSProp optimizer entry missing parameters; aborting load." << std::endl;
			return;
		}
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double rho = config.optimizer_params[2];
        double eps = config.optimizer_params[3];
        optimizer_ = std::make_unique<RMSProp>(lr, decay, eps, rho);
    }

    if (config.accuracy_type == "AccuracyCategorical") 
	{
        accuracy_ = std::make_unique<AccuracyCategorical>();
    } 
	else if (config.accuracy_type == "AccuracyRegression") 
	{
        accuracy_ = std::make_unique<AccuracyRegression>();
        if (!config.accuracy_params.empty()) 
		{
        }
    }

	if (layers_.empty())
	{
		std::cerr << "Model::LoadModel warning: no valid layers constructed; keeping model empty." << std::endl;
		return;
	}

	Finalize();

	if (config.parameters.size() != trainable_layers_.size())
	{
		std::cerr << "Model::LoadModel warning: parameter count mismatch; model left uninitialized." << std::endl;
		return;
	}

	SetParameters(config.parameters);
}

void NEURAL_NETWORK::Model::forward(const Eigen::MatrixXd& inputs, bool training)
{
	input_layer_->forward(inputs, training);
	bool flattened = false;
	
	for (const auto& layer : layers_)
	{
		auto prev = layer->getPrev();
		if (prev)
		{
			layer->forward(prev->GetOutput(), training);
		}
		else
		{
			layer->forward(inputs, training);
		}
	}

	output_ = layers_.back()->GetOutput();
}

void NEURAL_NETWORK::Model::backward(const Eigen::MatrixXd& output, 
									 const Eigen::MatrixXd& targets)
{
	auto start_iter = layers_.rbegin();
	if (softmax_classifier_)
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
			mse->backward(output, targets);
			layers_.back()->backward(mse->GetDInput());
		} 
		else
		{
			loss_->backward(output, targets);
			layers_.back()->backward(loss_->GetDInput());
		}
		start_iter++;
	}
	
	for (auto layer = start_iter; layer != layers_.rend(); layer++)
	{
		auto next = (*layer)->getNext();
		if (next)
		{
			(*layer)->backward(next->GetDInput());
		}
		else if (loss_)
		{
			(*layer)->backward(loss_->GetDInput());
		}
	}
}