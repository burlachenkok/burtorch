#include "RandomVariable.h"
#include <math.h>

namespace
{
    constexpr double kDoubleEps = 1e-10;
    constexpr double kPi = 3.14159265359;
    constexpr double k2Pi = 3.14159265359 * 2;
}

namespace burt
{
    double RandomVariable::generateUniform(double a, double b)
    {
        return uniformGen.generateReal(a, b);
    }

    double RandomVariable::generateExp(double lambda)
    {
        return -1 / lambda * log(1.0 - uniformGen.generateReal(0.0, 1.0 - kDoubleEps));
    }

    double RandomVariable::generateRayleigh(double sigma)
    {
        return sigma * sqrt(-2 * log(1.0 - uniformGen.generateReal(0.0, 1.0 - kDoubleEps)));
    }

    double RandomVariable::generateRayleigh()
    {
        return sqrt(-2 * log(1.0 - uniformGen.generateReal(0.0, 1.0 - kDoubleEps)));
    }

    double RandomVariable::generateNorm(double m, double sigma)
    {
        // Book: XVI Theory of Probability BMSTU 2004, p.247,248
        const double y1 = generateRayleigh();
        const double y2 = generateUniform(0.0, k2Pi);
        const double x1 = y1 * cos(y2);

        return sigma * x1 + m;
    }

    double RandomVariable::generateNorm()
    {
        // Book: XVI Theory of Probability BMSTU 2004, p.247,248
        const double y1 = generateRayleigh();
        const double y2 = generateUniform(0.0, k2Pi);
        const double x1 = y1 * cos(y2);
        return x1;
    }

    void RandomVariable::generateNorm2D(double& X1, double& X2)
    {
        // Book: XVI Theory of Probability BMSTU 2004, p.247,248
        double y1 = generateRayleigh();
        double y2 = generateUniform(0.0, k2Pi);
        X1 = y1 * cos(y2);
        X2 = y1 * sin(y2);
    }

    double RandomVariable::generateHiSquare(int n)
    {
        double r = double();
        for (int i = 0; i < n; ++i)
        {
            double x = generateNorm();
            r += x*x;
        }
        return r;
    }
}
