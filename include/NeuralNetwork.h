#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

// Layers
#include "LayerBase.h"
#include "LayerInput.h"
#include "LayerDense.h"
#include "LayerDropout.h"

// Activations
#include "ActivationLinear.h"
#include "ActivationReLU.h"
#include "ActivationSigmoid.h"
#include "ActivationSoftmax.h"
#include "ActivationSoftmaxLossCategoricalCrossentropy.h"

//Losses
#include "Loss.h"
#include "LossBinaryCrossEntropy.h"
#include "LossCategoricalCrossentropy.h"
#include "LossMeanAbsoluteError.h"
#include "LossMeanSquaredError.h"

// Accuracy
#include "Accuracy.h"
#include "AccuracyCategorical.h"
#include "AccuracyRegression.h"

//Optimizers
#include "Optimizer.h"
#include "AdaGrad.h"
#include "Adam.h"
#include "RMSProp.h"
#include "StochasticGradientDescent.h"

//Utils
#include "Helpers.h"
#include "Serialization.h"
#include "ZipReader.h"

#endif // __NEURAL_NETWORK_H__