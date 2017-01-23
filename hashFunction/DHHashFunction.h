#ifndef DOUBLEHADAMARDHASHFUNCTION_H
#define DOUBLEHADAMARDHASHFUNCTION_H

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
#include "DoubleHadamardFuncs.h"

class DoubleHadamardHashFunction {
    //subsampling - v. basic
	private:
        arma::uvec singleMul, subsample;

	public:

		DoubleHadamardHashFunction(void) {};

		//this constructor is used by RDHF
		DoubleHadamardHashFunction(const arma::uvec& tform,const uint64_t& nBits) {
		    subsample = tform;

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
				singleMul[i] = 1ull << (i);
			}
		}

		DoubleHadamardHashFunction(const uint64_t& nDims, const uint64_t& nBits) {

            subsample = arma::linspace<arma::uvec>(0,nDims-1,nDims);
            subsample = arma::shuffle(subsample);
            subsample = subsample.rows(0,nBits-1);

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
				singleMul[i] = 1ull << (i);
			}
		}

        const arma::uvec getHash(const arma::umat& d) const {
			return (d.rows(subsample).t() * singleMul);
		}

        const arma::uvec getHash(const arma::mat& d) const {
            return (arma::conv_to<arma::umat>::from(d.rows(subsample)).t() * singleMul);
        }

        const arma::uvec getHash(const arma::Mat<float>& d) const {
            return (arma::conv_to<arma::umat>::from(d.rows(subsample)).t() * singleMul);
        }
};

#endif
