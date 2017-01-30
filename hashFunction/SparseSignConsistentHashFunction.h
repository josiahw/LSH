#ifndef SPARSESIGNCONSISTENTHASHFUNCTION_H
#define SPARSESIGNCONSISTENTHASHFUNCTION_H

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

class SparseSignConsistentHashFunction {
    //random projection hash function
    private:
        std::vector<arma::uvec> sumIndices;
        arma::uvec singleMul;

    public:
        SparseSignConsistentHashFunction(void) {};

        SparseSignConsistentHashFunction(const std::vector<arma::uvec>& sumInds,const uint64_t& nBits) {
            //this constructor is used by RDHF
            sumIndices = sumInds;

            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        SparseSignConsistentHashFunction(const uint64_t& nDims, const uint64_t& nBits) {
            arma::uvec subsample = arma::linspace<arma::uvec>(0,nDims-1,nDims);
            //subsample = subsample.rows(0,nBits-1);
            //XXX: don't know the value of m yet - needs mathing from nDims
            size_t m = 5;
            while (sumIndices.size() < nBits) {
                subsample = arma::shuffle(subsample);
                arma::uvec candidates = subsample.rows(0,m);
                sumIndices.push_back(candidates(arma::find(arma::randn(m) > 0)));
                if (sumIndices.back().n_elem == 0) {
                    sumIndices.resize(sumIndices.size()-1);
                }
            }



            singleMul.resize(nBits);
            for (uint64_t i = 0 ; i < nBits; ++i) {
                singleMul[i] = 1ull << (i);
            }
        }

        const arma::uvec getHash(const arma::mat& d) const {
            arma::uvec output(d.n_cols,arma::fill::zeros);
            for (size_t i = 0; i < sumIndices.size(); ++i) {
                output += (singleMul[i] * (arma::sum(d.rows(sumIndices[i]),0) > 0)).t();
            }
            return output;
        }

        const arma::uvec getHash(const arma::Mat<float>& d) const {
            arma::uvec output(d.n_cols,arma::fill::zeros);
            for (size_t i = 0; i < sumIndices.size(); ++i) {
                output += (singleMul[i] * (arma::sum(d.rows(sumIndices[i]),0) > 0)).t();
            }
            return output;
        }
};

#endif
