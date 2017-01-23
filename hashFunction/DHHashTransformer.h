#ifndef DHHASHTRANSFORMER_H
#define DHHASHTRANSFORMER_H

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

class DHHashTransformer {
    private:
        arma::vec D; //bernoulli vector
        arma::vec G; //normal distribution vector
        arma::uvec M; //permutation vector
        arma::rowvec means;
    public:

        DHHashTransformer(void) {};

        DHHashTransformer(const arma::mat& data, const uint64_t& numDims) {

            means = arma::mean(data,0);

            //get the power of 2 size required
            int size = 1;
            while (size < data.n_cols) {
                size *= 2;
            }

            //get the random variables needed for hadamard hashing
            G.randn(size/2);
            D.randn(size);
            D = 2.*arma::conv_to<arma::vec>::from(D > 0.) - 1.;
            M = arma::linspace<arma::uvec>(0,size-1,size);
            M = arma::shuffle(M);
            M = M.rows(0,size/2-1);
        }

        arma::umat transform(const arma::mat& d) {
            return HadamardRecursive<arma::mat>(G,
                                    HadamardRecursive<arma::mat>(D,d,means).cols(M)
                                    ) > 0;
        }

        arma::umat transform(const arma::Mat<float>& d) {
            return HadamardRecursive<arma::Mat<float>>(G,
                                    HadamardRecursive<arma::Mat<float>>(D,d,means).cols(M)
                                    ) > 0;
        }
};

#endif