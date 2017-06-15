#ifndef RANDOMSUBSAMPLINGHASHFUNCTION_H
#define RANDOMSUBSAMPLINGHASHFUNCTION_H

#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <list>
#include <iostream>

class RandomSubSamplingHashFunction {
    //random rotation hash function - v. basic
    private:
        arma::mat transform;
        arma::Mat<float> transformf;
        arma::uvec singleMul, subsample;


    public:

        RandomSubSamplingHashFunction(void) {};

        //this constructor is used by random subsampling
        RandomSubSamplingHashFunction(const arma::uvec& tform,const uint64_t& nBits) {
            subsample = tform;

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        RandomSubSamplingHashFunction(const uint64_t& nDims, const uint64_t& nBits) {

            subsample = arma::linspace<arma::uvec>(0,nDims-1,nDims);
            subsample = arma::shuffle(subsample);
            subsample = subsample.rows(0,nBits-1);

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        const arma::uvec getHash(const arma::umat& d) const {
            return (singleMul.t() * d.rows(subsample)).t();
        }
};



#endif
