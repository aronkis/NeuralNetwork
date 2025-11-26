#ifndef __PLUTO_TRAIN_H__
#define __PLUTO_TRAIN_H__

#include <iostream>
#include <set>
#include "Model.h"
#include "Common.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define SAVE_EVERY 0

using Symbol = std::pair<double, double>;

int pluto_train_main();

#endif // __PLUTO_TRAIN_H__