#ifndef RDHFCONSTRUCTOR_H
#define RDHFCONSTRUCTOR_H
#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <list>
#include "HashError.h"

template<class DataTransformer, class HashFunction, class HashIndex, class HashCollection, class MAT>
class RDHFConstructor {
private:
    struct SparseMat {
        arma::uvec indices;
        arma::vec values;
    };

    struct HashFuncData {
        arma::vec means;
        arma::mat projections;
    };

    static HashFuncData genHashFuncs (const arma::mat& data, const uint64_t numBits) {
        arma::arma_rng::set_seed_random();
        #ifdef PCARR
        arma::mat result(numBits,data.n_cols);
        uint64_t count = 0;
        while (count < numBits) {
            arma::mat tmp1;
            arma::mat tmp0;
            arma::vec tmp2;
            arma::svd(tmp0,tmp2,tmp1,arma::randn(data.n_cols,data.n_cols));
            uint64_t last = std::min<uint64_t>(count+data.n_cols,result.n_rows);
            result.rows(count,last-1) = tmp0.rows(0,last-count-1);
            count += data.n_cols;
        }


        return {arma::mean(data,0).t(),result};
        #else

        #ifdef RANDOMSUBSAMPLING
        return {arma::mean(data,0).t(),arma::randn(numBits,data.n_cols)};
        #else
        //return random projections
        return {arma::mean(data,0).t(),arma::randn(numBits,data.n_cols) * HashFunction::SIKH_CONST};
        #endif
        #endif
    }

    static SparseMat getSMatrix(const arma::umat& neighbourData,
                         const uint64_t numNeighbours,
                         const uint64_t numNonNeighbours,
                         const uint64_t matrixSize) {
        //gets a sparse version of the similarity matrix
        arma::uvec indices( neighbourData.n_rows * (numNeighbours + numNonNeighbours) );
        arma::vec weights( neighbourData.n_rows * (numNeighbours + numNonNeighbours) );
        //arma::uvec candidates = arma::linspace<arma::uvec>(0,matrixSize-1,matrixSize);
        //#pragma omp parallel for
        for (uint64_t i = 0; i < neighbourData.n_rows; ++i) {
            for (uint64_t j = 0; j < numNeighbours; ++j) {
                indices[(numNeighbours + numNonNeighbours)*i + j] = matrixSize*i + neighbourData(i,j);
                weights[(numNeighbours + numNonNeighbours)*i + j] = 1.;
            }
            uint64_t l = 0, k = 0;
            arma::arma_rng::set_seed_random();

            arma::uvec candidates2 = arma::randi<arma::uvec>(2*numNonNeighbours,arma::distr_param(0,2*numNonNeighbours)); //arma::shuffle(candidates);
            std::set<uint64_t> trueneighbours;
            for (uint64_t j = 0; j < numNeighbours; ++j) {
                trueneighbours.insert(neighbourData(i,j));
            }
            while (l < numNonNeighbours and k < candidates2.n_elem) {
                if ( trueneighbours.count(candidates2[k])==0 ) {
                    indices[(numNeighbours + numNonNeighbours)*i + l + numNeighbours] = matrixSize*i + candidates2[k];
                    weights[(numNeighbours + numNonNeighbours)*i + l + numNeighbours] = -1.;
                    ++l;
                    trueneighbours.insert(candidates2[k]);
                }
                ++k;

            }
        }
        return {indices,weights};
    }

    static arma::mat getYMatrix(arma::mat data, const HashFuncData& hashFuncs) {
        data.each_row() -= hashFuncs.means.t();
        #ifdef SIKH
        data = 2.*arma::conv_to<arma::mat>::from(arma::sin(hashFuncs.projections * data.t()) > 0.) - 1.;
        #else
        data = 2.*arma::conv_to<arma::mat>::from(hashFuncs.projections * data.t() > 0.) - 1.;
        #endif
        return data;
    }

    static arma::vec getPiVector(const arma::mat& yMatrix, const SparseMat& sMatrix, double gamma) {
        //Process Y * S * Y^T
        arma::vec pi = arma::zeros(yMatrix.n_rows);
        #pragma omp parallel for
        for (uint64_t k = 0; k < yMatrix.n_rows; ++k) {
            uint64_t j = 0;
            for (uint64_t i = 0; i < yMatrix.n_cols and j < sMatrix.indices.n_elem; ++i) {
                double result = 0.;
                while (sMatrix.indices[j] < (i+1)*yMatrix.n_cols and j < sMatrix.indices.n_elem) {
                    result += (yMatrix(k,sMatrix.indices[j] - i*yMatrix.n_cols) * sMatrix.values[j]);
                    ++j;
                }
                pi[k] += result * yMatrix(k,i);

            }
        }
        pi /= yMatrix.n_rows*500; //scale pi to be the same as the original paper
        //std::cout << pi << std::endl;
        return arma::exp(-gamma*pi); //NOTE: it seems that this should have a - sign here. Not detailed in the original papers but necessary to make it optimise upward.
    }

    static arma::mat getAMatrix(const arma::mat& yMatrix, const double lamda) {
        arma::vec pr1 = arma::mean(yMatrix,1)*0.5 + 0.5;
        arma::vec pr0 = 1.0 - pr1;

        //1-1 case
        arma::mat pr = pr1 * pr1.t() + 0.00001;
        arma::mat prab = (yMatrix+1.)*(yMatrix+1.).t()/(4.*yMatrix.n_cols);
        arma::mat result = prab % arma::log( prab / pr );

        //1-0 case
        pr = pr1 * pr0.t() + 0.00001;
        prab = (yMatrix+1.)*(1.-yMatrix).t()/(4.*yMatrix.n_cols);
        result += prab % arma::log( prab / pr );

        //0-1 case
        pr = pr0 * pr1.t() + 0.00001;
        prab = (1.-yMatrix)*(yMatrix+1.).t()/(4.*yMatrix.n_cols);
        result += prab % arma::log( prab / pr );

        //0-0 case
        pr = pr0 * pr0.t() + 0.00001;
        prab = (1.-yMatrix)*(1.-yMatrix).t()/(4.*yMatrix.n_cols);
        result += prab % arma::log( prab / pr );

        //these need to be zeroed out so as not to cause issues later on
        for (uint64_t i = 0; i < result.n_cols; ++i) {
            result(i,i) = 0.;
        }

        return arma::exp(-lamda*result);
    }

    static arma::mat getAHatMatrix(const arma::mat& aMatrix, const arma::vec& pi) {
        return arma::diagmat(pi) * aMatrix * arma::diagmat(pi);
    }

    static arma::uvec getZStar(arma::mat aHat, const uint64_t& numBits) {
        arma::uvec results(numBits);
        uint64_t i = 0;
        //std::cout << aHat << std::endl;
        arma::uvec candidates = arma::linspace<arma::uvec>(0,aHat.n_cols-1,aHat.n_cols);
        while (i < numBits) {
            arma::vec z = arma::ones(aHat.n_cols)/aHat.n_cols;
            z = (aHat.t() * z) / arma::accu( (aHat.t() * z) % z);
            for (uint64_t j = 0; j < 1000; ++j) {
                arma::vec tmp = (aHat.t() * z) / arma::accu( (aHat.t() * z) % z);
                tmp = (aHat.t() * tmp) / arma::accu( (aHat.t() * tmp) % tmp); //undocumented in the paper: the answer oscillates between 2 values, so 2 steps makes sure we always converge
                double diff = arma::norm(z - tmp,2);
                z = tmp;
                if (diff < 2.0e-18) { //call me picky but there is definitely improvement in optimisation by going to e-18 - maybe in some cases it changes the ordering of choices,
                    break;
                }
            }
            arma::uvec sorted = arma::flipud(arma::sort_index(z)); //1 indicates descending sort
            uint64_t i0 = i; //this is the indexing offset
            if (z[sorted[0]] > 0.01/z.n_elem) { //we have found a definite convergence, add it to our hash function

                while (i < numBits && z[sorted[i-i0]] > 0.01/z.n_elem) {
                    results[i] = candidates[sorted[i-i0]];
                    ++i;
                }
                candidates = candidates.rows(sorted.rows(i-i0+1,sorted.n_rows-1));
                aHat = aHat.rows(sorted.rows(i-i0+1,sorted.n_rows-1));
                aHat = aHat.cols(sorted.rows(i-i0+1,sorted.n_rows-1));

            } else { //this indicates that there are way too many candidates or we haven't converged properly, so we take the best few channels
                while (i < numBits) {
                    results[i] = candidates[sorted[i-i0]];
                    ++i;
                }
            }

        }

        return results;
    }

    static arma::vec getPMatrix(const arma::mat& data,
                         const arma::mat& queries,
                         const SparseMat& sMatrix,
                         const HashFuncData& lastHashFunc) {
        //we keep track of the minimum Hamming distances coinciding with our sparse matrix S.
        static arma::vec distances(sMatrix.indices.n_elem,arma::fill::zeros);

        if (lastHashFunc.means.n_elem > 0) {
            #pragma omp parallel for
            for (uint64_t i = 0; i < distances.n_elem; ++i) {
                uint64_t j = sMatrix.indices[i] % data.n_cols;
                uint64_t k = sMatrix.indices[i] / data.n_cols;
                arma::mat c1 = data.col(j);
                arma::mat c2 = queries.col(k);
                distances[i] = std::min<double>(distances[i],
                                     arma::norm(
                                                (c1-c2)/4.,
                                               int(1)));
            }
        } else {
            distances += 64.;
        }
        const double u = arma::accu( distances(arma::find(sMatrix.values > 0.)) )/arma::accu(sMatrix.values>0.);

        arma::vec modDists = distances;
        if (lastHashFunc.means.n_elem == 0) {
            modDists *= 0.;
            modDists += 0.00000000000000000000000000000001;
        } else {
            modDists -= u;
        }
        return modDists;
    }

    static SparseMat getWMatrix(const SparseMat& sMatrix, const arma::vec& pMatrix) {

        arma::vec vals = pMatrix % arma::sign(sMatrix.values);
        double alpha = arma::accu(vals < 0.) / (arma::accu(vals >0.0) + 0.000001);
        vals = arma::exp(-alpha * vals) % sMatrix.values;
        vals -= arma::conv_to<arma::vec>::from(vals < -50.) % (vals + 50.) ;
        vals -= arma::conv_to<arma::vec>::from(vals > 50.) % (vals - 50.) ; //this stops infinities from badly conditioned weights - it's more effective than reweighting
        vals -= arma::conv_to<arma::vec>::from(vals == 0.0)*0.000000001; //this helps stop infinities from log(0), but the algorithm usually works without it.
        return {sMatrix.indices,vals};
    }
public:
    static HashCollection GetHashes(const arma::Mat<uint32_t>& queries,
                                    MAT& data,
                                    uint64_t bitPoolSize,
                                    uint64_t numFuncs,
                                    uint64_t numBits,
                                    uint64_t minquery,
                                    uint64_t querydepth,
                                    uint64_t querysize,
                                    uint64_t PCAbits = 128,
                                    double gamma = 0.2,
                                    double lamda = 4.) {
        //make a hash collection
        HashCollection results(data,PCAbits,querysize,minquery,querydepth,numBits);
        std::cout << "RDHF" << std::endl;
        //initialise the once-off variables
        SparseMat S = getSMatrix(arma::conv_to<arma::umat>::from(queries),
                                queries.n_cols,
                                2*queries.n_cols,
                                data.n_rows);
        HashFuncData hf = genHashFuncs(data, bitPoolSize);
        arma::mat Y = getYMatrix(data,hf);
        arma::mat A = getAMatrix(Y,lamda);
        arma::mat lastHashY;

        //init last hash func
        HashFuncData lastHash; //(arma::zeros<arma::vec>(0),arma::mat(0));

        for (uint64_t i = 0; i < numFuncs; ++i) {
            //std::cout << i << std::endl;
            S = getWMatrix(S,getPMatrix(lastHashY,lastHashY,S,lastHash));
            //std::cout << S.indices << std::endl;
            arma::vec pi = getPiVector(Y,S,gamma);
            //std::cout << pi << std::endl;
            arma::uvec candidates = getZStar(getAHatMatrix(A,pi),numBits);
            pi.clear();
            //collect last hash function
            lastHash.means = hf.means;
            lastHash.projections = hf.projections.rows(candidates);

            //add to the hash collection
            results.addHash( HashFunction(lastHash.projections,numBits) );

            //shrink active matrices to exclude the generated function
            arma::uvec newIndices(A.n_cols - numBits);
            uint64_t k = 0;
            for (uint64_t j = 0; j < A.n_cols; ++j) {
                if (arma::accu(candidates == j) == 0) {
                    newIndices[k] = j;
                    ++k;
                }
            }
            hf.projections = hf.projections.rows(newIndices);
            A = A.cols(newIndices);
            A = A.rows(newIndices);
            lastHashY = Y.rows(candidates);
            Y = Y.rows(newIndices);
        }
        return results;
    }

};

#endif