#ifndef __PLUTO_EVAL_H__
#define __PLUTO_EVAL_H__

#include <iostream>
#include <set>
#include "Model.h"
#include "Common.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Symbol = std::pair<double, double>;

int pluto_eval_main();

#endif // __PLUTO_EVAL_H__