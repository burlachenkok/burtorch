/** @file
* Emulate r.v. with specific distribution
*/
#pragma once

#include "RandomGenMersenne.h"

namespace burt
{
    /** Collection of generators of r.v. with specific distribution
    */
    class RandomVariable
    {
    public:
        /** Generate  r.v. with uniform distribution, and mx=(a+b)/2, dx=(b-a)^2 / 12
        * @remark p.d.f. has form: f(x)=1/b-a \cdot I(x \in [a,b])
        */
        double generateUniform(double a = 0.0, double b = 1.0);

        /** Generate r.v. with exp distribution mx=lambda^-1, dx=lambda^-2
        * @remark p.d.f. has form: f(x)=lambda \cdot \exp(-\lambda x)
        */
        double generateExp(double lambda = 1.0);

        /** Generate r.v. with Rayleigh distribution mx=(pi/2)^0.5*sigma, dx=(2-pi/2)*sigma^2
        * If point in the plane has (vx, vy) s.t. vx ~ N, vy ~ N => |(vx, vy)| ~ Has Relei distribution
        */
        double generateRayleigh(double sigma);

        /** Generate r.v. with Rayleigh distribution mx=(pi/2)^0.5*sigma, dx=(2-pi/2)*sigma^2 with parameter sigma = 1.0
        * If point in the plane has (vx, vy) s.t. vx ~ N, vy ~ N => |(vx, vy)| ~ Has Relei distribution
        */
        double generateRayleigh();

        /** Generate r.v. with norm distribution
        * @param m mean
        * @param sigma standart deviation
        */
        double generateNorm(double m, double sigma);

        /** Generate r.v. with norm distribution with mean equal 0 and variance equal 1.
        */
        double generateNorm();

        /** Generate 2d r.v. with standard norm distribution. i.e. p.d.f. is (1/2pi)*exp(-1/2 (x1^2+x2^2))
        */
        void generateNorm2D(double& X1, double& X2);

        /** Generate r.v. with HiSquare(n) distribution
        */
        double generateHiSquare(int n = 1);

    private:
        RandomGenMersenne uniformGen; ///< Under the hood generator for r.v.
    };
}
