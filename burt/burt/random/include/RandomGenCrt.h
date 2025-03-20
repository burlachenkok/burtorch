/** @file
* Emulate i.r.v via standard random generator
*/

#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

namespace burt
{
    /** Emulate i.r.v via standard random generator
    */
    class RandomGenCrt
    {
    public:
        /** Get reference to some global instance of generator
        * @return global instance of this generator
        */
        static RandomGenCrt& global();

        /** Ctor. Initialize generator to generate random values in [A,B]
        * @param A lower bound of interval to generate numbers
        * @param B upper bound of interval to generate numbers
        */
        RandomGenCrt(double A = 0.0, double B = 1.0);

        /** Get lower bound
        * @return lower bound
        */
        double getA() const;
        
        /** Get upper bound of generator
        * @return lower bound of generator
        */
        double getB() const;

        /** Generate pseudo random number in [0, 1]
        * @return generated
        */
        template <class TFloatType = double>
        TFloatType generateRealInUnitInterval()
        {
            const double val = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            genlast = val;
            return (TFloatType)genlast;
        }
        
        /** Generate batch of pseudo random number in [0,1]
        * @param placeHolder result array
        * @param kNumbers numbers to generate
        * @return nothing
        */
        template <class TFloatType>
        void generateBatchOfRealsInUnitInterval(TFloatType* placeHolder, size_t kNumbers)
        {
            for (size_t i = 0; i < kNumbers; ++i)
                placeHolder[i] = generateRealInUnitInterval<TFloatType>();
            
            return;
        }
        
        /** Generate pseudo random number
        * @return generated
        */
        double generateReal();
        
        /** Obtain information about last generate pseudo random number
        * @return generated
        */
        double last() const;

        /** Internal seed for random generator
        * @return seed value
        */
        static uint64_t getSeed();

        /** Set seed as static function because CRT do not give many functions to operate on it.
        * @param seed used seed to initialize random generator
        */
        static void setSeed(uint64_t seed);


    private:
        static uint64_t seed; ///< Global seed
        double genlast;       ///< Instance of last generated r.v.
        double sA;            ///< A for [A,B] interval in which r.v. are sampled uniformly
        double sB;            ///< B for [A,B] interval in which r.v. are sampled uniformly
    };
}
