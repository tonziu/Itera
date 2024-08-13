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

    double map_value(double value, double in_min, double in_max, double out_min, double out_max)
    {
        assert(in_min < in_max);
        assert(out_min < out_max);

        double scaled_value = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min;

        if (scaled_value < out_min)
            return out_min;
        else if (scaled_value > out_max)
            return out_max;

        return scaled_value;
    }
}

#endif // H_ITERA_FUNCTIONS_H