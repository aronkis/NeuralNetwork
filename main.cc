#include "Common.h"

int main(int argc, char **argv)
{
#if defined(NEURAL)
	return neural_train_main();
#endif

#if defined(NEURAL_MODEL)
	return neural_eval_main(argc, argv);
#endif

#if defined(CNN)
	return cnn_train_main();
#endif

#if defined(CNN_MODEL)
	return cnn_eval_main();
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