#ifndef RANDOMCONSTRUCTOR_H
#define RANDOMCONSTRUCTOR_H

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

#include <iostream>

template<class DataTransformer, class HashFunction, class HashIndex, class HashCollection, class MAT>
class RandomConstructor {
    private:

    public:

        static HashCollection GetHashes(
                            const arma::Mat<uint32_t>& GroundTruth,
                            const MAT& BaseData, //groundtruth are assumed to be the first x values in base data
                            const uint64_t& candidateHashFunctions,
                            const uint64_t& totalHashFunctions,
                            const uint64_t& minRetrievalThreshold,
                            const uint64_t& maxSearchThreshold, // this is the hamming ball distance or tree level distance to search
                            const uint64_t& maxQuerySize,
                            const uint64_t& hashBits) {
                //make a new hash-function collection
                //arma::mat _data = arma::conv_to<arma::mat>::from(BaseData.getData());
                HashCollection results(BaseData,
                                        hashBits,
                                        maxQuerySize,
                                        minRetrievalThreshold,
                                        maxSearchThreshold,
                                        hashBits);
                //PCA-transform the base-data
                auto data = results.transform(BaseData);

                for (uint64_t i = 0; i <= totalHashFunctions; ++i) {
                    arma::arma_rng::set_seed_random();
                    results.addHash(HashFunction(data.n_rows,hashBits));
                }
                return results;
            }

};



#endif
