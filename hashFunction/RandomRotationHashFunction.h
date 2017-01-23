#ifndef RANDOMROTATIONHASHFUNCTION_H
#define RANDOMROTATIONHASHFUNCTION_H

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

class RandomRotationHashFunction {
    //random rotation hash function - v. basic
	private:
		arma::mat transform;
		arma::Mat<float> transformf;
        arma::uvec singleMul, subsample;


	public:
	    constexpr static double SIKH_CONST = 0.8;

		RandomRotationHashFunction(void) {}

		RandomRotationHashFunction(const arma::mat& tform,const uint64_t& nBits) {
		    transform = tform;
		    transformf = arma::conv_to<arma::Mat<float>>::from(transform);

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
				singleMul[i] = 1ull << (i);
			}
		}

		RandomRotationHashFunction(const uint64_t& nDims, const uint64_t& nBits) {
		    arma::mat tmp1;
		    arma::vec tmp2;
		    arma::svd(transform,tmp2,tmp1,arma::randn(nDims,nDims)); //XXX: lazy, assume ndims is always bigger
		    if (nBits < nDims) {
		        transform.shed_cols(nBits,transform.n_rows-1);
		    }
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
