#ifndef RANDOMPROJECTIONHASHFUNCTION_H
#define RANDOMPROJECTIONHASHFUNCTION_H

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

class RandomProjectionHashFunction {
    //random projection hash function
    private:
        arma::mat transform;
        arma::Mat<float> transformf;
        arma::uvec singleMul;

    public:
        RandomProjectionHashFunction(void) {};

        RandomProjectionHashFunction(const arma::mat& tform,const uint64_t& nBits) {
            //this constructor is used by RDHF
            transform = tform;
            transformf = arma::conv_to<arma::Mat<float>>::from(transform);

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        RandomProjectionHashFunction(const uint64_t& nDims, const uint64_t& nBits) {
            transform = arma::randn(nBits,nDims);
            transformf = arma::conv_to<arma::Mat<float>>::from(transform);


            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        const arma::uvec getHash(const arma::mat& d) const {
            return (transform * d > 0.0).t() * singleMul;
        }

        const arma::uvec getHash(const arma::Mat<float>& d) const {
            return (transformf * d > 0.0).t() * singleMul;
        }
};

#endif
