#include "Model.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

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

	for (int i = 0; i < layer_count; ++i)
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
	}

	if (loss_)
	{
		std::vector<std::weak_ptr<LayerDense>> weak_trainables;
		weak_trainables.reserve(trainable_layers_.size());

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

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		std::cout << "Epoch: " << epoch << std::endl;
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
					optimizer_->UpdateParameters(*layer_sp);
				}
			}

			optimizer_->PostUpdateParameters();
			if (batch_size > 1 && 
				(!(step % print_every) || 
				 (step == train_steps - 1)))
			{
				std::cout << "Step: " << step 
						  << ", Accuracy: " << accuracy 
						  << ", Loss: " << loss 
						  << " (Data loss: " << loss_->GetLoss()
						  << " | Regularization loss: " << reg_loss
						  << "), Learning Rate: " << optimizer_->GetLearningRate()
						  << '\n';
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
	for (size_t i = 0; i < trainable_layers_.size(); ++i)
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
		else if (auto* dropout = dynamic_cast<LayerDropout*>(layer.get())) 
		{
            config.layer_types.push_back("LayerDropout");
            config.layer_params.push_back({dropout->GetRate()});
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

    if (auto* binary = dynamic_cast<LossBinaryCrossEntropy*>(loss_.get())) 
	{
        config.loss_type = "LossBinaryCrossEntropy";
        config.loss_params = {};
    } 
	else if (auto* categorical = dynamic_cast<LossCategoricalCrossEntropy*>(loss_.get())) 
	{
        config.loss_type = "LossCategoricalCrossEntropy";
        config.loss_params = {};
    } 
	else if (auto* mse = dynamic_cast<LossMeanSquaredError*>(loss_.get())) 
	{
        config.loss_type = "LossMeanSquaredError";
        config.loss_params = {};
    } 
	else if (auto* mae = dynamic_cast<LossMeanAbsoluteError*>(loss_.get())) 
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

    if (auto* cat = dynamic_cast<AccuracyCategorical*>(accuracy_.get())) 
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

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
	{
		std::cerr << "Failed to open file for writing\n";
		return;
	}

    size_t numLayers = config.layer_types.size();
    NEURAL_NETWORK::Serialization::WriteRaw(ofs, &numLayers, sizeof(numLayers));
    
	for (size_t i = 0; i < numLayers; ++i) 
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
	std::cout << "Model saved to " << path << std::endl;
}

void NEURAL_NETWORK::Model::LoadModel(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
	{
		std::cerr << "Failed to open file for reading\n";
		return;
	}

    ModelConfig config;

    size_t numLayers;
    NEURAL_NETWORK::Serialization::ReadRaw(ifs, &numLayers, sizeof(numLayers));
    
	config.layer_types.resize(numLayers);
    config.layer_params.resize(numLayers);

    for (size_t i = 0; i < numLayers; ++i) 
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

    softmax_classifier_ = config.softmax_classifier_;

    layers_.clear();
    trainable_layers_.clear();
    loss_.reset();
    optimizer_.reset();
    accuracy_.reset();

    for (size_t i = 0; i < config.layer_types.size(); ++i) 
	{
        const auto& type = config.layer_types[i];
        const auto& params = config.layer_params[i];

        if (type == "LayerInput") 
		{
            Add(std::make_shared<LayerInput>());
        } 
		else if (type == "LayerDense") 
		{
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
            double rate = params[0];
            Add(std::make_shared<LayerDropout>(rate));
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
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double b1 = config.optimizer_params[2];
        double b2 = config.optimizer_params[3];
        double eps = config.optimizer_params[4];
        optimizer_ = std::make_unique<Adam>(lr, decay, b1, b2, eps);
    } 
	else if (config.optimizer_type == "StochasticGradientDescent") 
	{
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double momentum = config.optimizer_params[2];
        optimizer_ = std::make_unique<StochasticGradientDescent>(lr, decay, momentum);
    } 
	else if (config.optimizer_type == "AdaGrad") 
	{
        double lr = config.optimizer_params[0];
        double decay = config.optimizer_params[1];
        double eps = config.optimizer_params[2];
        optimizer_ = std::make_unique<AdaGrad>(lr, decay, eps);
    } 
	else if (config.optimizer_type == "RMSProp") 
	{
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

    Finalize();
    SetParameters(config.parameters);
}

void NEURAL_NETWORK::Model::forward(const Eigen::MatrixXd& inputs, bool training)
{
	input_layer_->forward(inputs, training);
	
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
	
	for (auto layer = start_iter; layer != layers_.rend(); ++layer)
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