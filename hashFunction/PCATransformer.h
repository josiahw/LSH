#ifndef PCATRANSFORMER_H
#define PCATRANSFORMER_H

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

class PCATransformer {
    /*
        A pre-LSH low-dimensional projection filter. This allows hash function collections
        to specialise a common pre-filtering step to be done once for hte whole collection.
        TODO: figure out a way to make this nicer, and rename it
    */
    private:
        arma::mat transformMatrix;
        arma::Mat<float> transformMatrixf;
        arma::rowvec means;
    public:

        PCATransformer(void) {};

        PCATransformer(const arma::mat& data, const uint64_t& numDims) {

            means = arma::mean(data,0);
            auto d = data;
            d.each_row() -= means;
            arma::princomp(transformMatrix,d);
            if (numDims < data.n_cols) {
                transformMatrix.shed_cols(numDims,transformMatrix.n_cols-1);
            }
            transformMatrixf = arma::conv_to<arma::Mat<float>>::from(transformMatrix);
        }

        arma::mat transform(const arma::mat& d) {
            return arma::mat(transformMatrix.t() *
                        (d - arma::repmat(means.t(),1,d.n_cols))
                );
        }

        arma::Mat<float> transform(const arma::Mat<float>& d) {
            return arma::Mat<float>(transformMatrixf.t() *
                        (d - arma::repmat( arma::conv_to<arma::Col<float_t>>::from(means.t()) ,1,d.n_cols))
                );
        }
};

#endif

