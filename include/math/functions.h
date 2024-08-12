#ifndef H_ITERA_FUNCTIONS_H
#define H_ITERA_FUNCTIONS_H

#include <cmath>

namespace math
{
    double relu(double x)
    {
        return x > 0 ? x : 0.0;
    }

    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double tanh(double x)
    {
        return std::tanh(x);
    }
}

#endif // H_ITERA_FUNCTIONS_H