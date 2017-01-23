#ifndef SHIFTINVARIANTKERNELHASHFUNCTION_H
#define SHIFTINVARIANTKERNELHASHFUNCTION_H

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

class ShiftInvariantKernelHashFunction {
    //SIKH hash function
    private:
        arma::mat transform;
        arma::Mat<float> transformf;
        arma::uvec singleMul;

    public:
        //XXX: fix this to be configurable
        constexpr static double SIKH_CONST = 0.8;

        ShiftInvariantKernelHashFunction(void) {};

        ShiftInvariantKernelHashFunction(const arma::mat& tform,const uint64_t& nBits) {
            //this constructor is used by RDHF
            transform = tform;
            transformf = arma::conv_to<arma::Mat<float>>::from(transform);

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        ShiftInvariantKernelHashFunction(const uint64_t& nDims, const uint64_t& nBits) {
            transform = arma::randn(nBits,nDims);
            transform *= SIKH_CONST;
            transformf = arma::conv_to<arma::Mat<float>>::from(transform);


            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        /*const uint64_t getHash(const arma::rowvec& d) const {
            return arma::dot(arma::sin(d.t() * transform) > 0.0, singleMul);
        }

        const uint64_t getHash(const arma::Row<float>& d) const {
            return arma::dot(arma::sin(d.t() * transformf) > 0.0, singleMul);
        }

        const uint64_t getHash(const arma::urowvec& d) const {
            return arma::dot(arma::sin(arma::conv_to<arma::Row<float>>::from(d) * transformf.t()) > 0.0, singleMul);
        }*/

        const arma::uvec getHash(const arma::mat& d) const {
            return (arma::sin(transformf * d) > 0.0).t() * singleMul;
        }

        const arma::uvec getHash(const arma::Mat<float>& d) const {
            return (arma::sin(transformf * d) > 0.0).t() * singleMul;
        }
};

#endif
