// #define PLUTO
// #define PLUTO_MODEL
// #define FASHION_MNIST
// #define MODEL
// #define CNN
// #define CNN_MODEL
// #define RADIO_1D_CNN
// #define GNU_MODEL
#define RADIO_1D_CNN_MODEL

#if defined(FASHION_MNIST)
#include "neural_train.h"
#endif

#if defined(MODEL)
#include "neural_eval.h"
#endif

#if defined(CNN)
#include "cnn_train.h"
#endif

#if defined(CNN_MODEL)
#include "cnn_eval.h"
#endif

#if defined(PLUTO)
#include "pluto_train.h"
#endif

#if defined(PLUTO_MODEL)
#include "pluto_eval.h"
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

int main(int argc, char **argv)
{
#if defined(FASHION_MNIST)
	return neural_train_main();
#endif

#if defined(MODEL)
	return neural_eval_main(argc, argv);
#endif

#if defined(CNN)
	return cnn_train_main();
#endif

#if defined(CNN_MODEL)
	return cnn_eval_main();
#endif

#if defined(PLUTO)
	return pluto_train_main();
#endif

#if defined(PLUTO_MODEL)
	return pluto_eval_main();
#endif

#if defined(RADIO_1D_CNN)
	return radio_train_main();
#endif

#if defined(RADIO_1D_CNN_MODEL)
	return radio_eval_main();
#endif

#if defined(GNU_MODEL)
	return gnu_eval_main();
#endif
}