#ifndef __COMMON_H__
#define __COMMON_H__

#include "Model.h"

#ifndef NN_EPOCHS
#define NN_EPOCHS 5
#endif

#ifndef NN_PRINT_EVERY
#define NN_PRINT_EVERY 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 32
#endif

#define NEURAL
// #define NEURAL_MODEL
// #define CNN
// #define CNN_MODEL
// #define RADIO_1D_CNN
// #define RADIO_1D_CNN_MODEL
// #define GNU_MODEL

#if defined(NEURAL)
#include "neural_train.h"
#endif

#if defined(NEURAL_MODEL)
#include "neural_eval.h"
#endif

#if defined(CNN)
#include "cnn_train.h"
#endif

#if defined(CNN_MODEL)
#include "cnn_eval.h"
#endif

#if defined(RADIO_1D_CNN)
#include "radio_train.h"
#endif

#if defined(RADIO_1D_CNN_MODEL)
#include "radio_eval.h"
#endif

#if defined(GNU_MODEL)
#include "gnu_eval.h"
#endif

#endif // __COMMON_H__