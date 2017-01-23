#ifndef BOOSTHASH_H
#define BOOSTHASH_H

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
class HashBooster {
	private:

	public:

		static HashCollection GetHashes(
		                    const arma::Mat<uint32_t>& GroundTruth,
		                    const MAT& BaseData, //groundtruth are assumed to be the first x values in base data
		                    const uint64_t& candidateHashFunctions,
		                    const uint64_t& totalHashFunctions,
		                    const uint64_t& minRetrievalThreshold,
		                    const uint64_t& maxSearchThreshold,
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
		        std::cout << "Boosting" << std::endl;
		        //PCA-transform the base-data
		        auto data = results.transform(BaseData);

		        //short circuit logic if we don't need to boost
		        if (candidateHashFunctions == 1) {
		            for (uint64_t i = 0; i <= totalHashFunctions; ++i) {
		                arma::arma_rng::set_seed_random();
		                results.addHash(HashFunction(data.n_cols,hashBits));
		            }
		            return results;
		        }

		        //initialise the boost-weights
		        arma::mat groundTruthWeights = arma::mat(GroundTruth.n_cols,GroundTruth.n_rows,arma::fill::ones);
		        std::vector<std::set<uint64_t>> groundTruthSets(groundTruthWeights.n_rows);
		        //fill the ground-truth sets
		        #pragma omp parallel for
		        for (uint64_t i = 0; i < groundTruthWeights.n_rows; ++i) {
					const arma::Col<uint32_t>& gt = GroundTruth.col(i);
		            groundTruthSets[i].insert(gt.begin(),gt.end());

		            //set the identity truths to 0
		            uint64_t j = 0;
		            for (const auto& s : groundTruthSets[i]) {
		                if (s == i) {
		                    groundTruthWeights(i,j) = 0.0;
		                    break;
		                }
		                ++j;
		            }
		        }

		        //set up our totals
		        double totalPairs = BaseData.n_cols * GroundTruth.n_cols - GroundTruth.n_cols;
		        double summedTruths = arma::accu(groundTruthWeights);

		        //set the F-weights
		        double fWeight = 1.0;
		        double totalfWeights = totalPairs - summedTruths;

		        //normalise the distribution
		        groundTruthWeights /= totalPairs;
		        fWeight /= totalPairs;
		        summedTruths /= totalPairs;

				//std::cout << "Ground Truths Processed, starting loop" << std::endl;

		        //create all candidate pre-calculations first and re-use every loop
		        std::vector<std::tuple<double,classificationErrors,HashFunction>> candidates(candidateHashFunctions);
		        #pragma omp parallel for
		        for (uint64_t j = 0; j < candidates.size(); ++j) {
		            arma::arma_rng::set_seed_random();
		            std::get<2>(candidates[j]) = HashFunction(data.n_cols,hashBits);
		            #ifdef RANDOMSUBSAMPLING
		            arma::uvec hashes = std::get<2>(candidates[j]).getHash(arma::umat(data.t()));
		            #else
		            arma::uvec hashes = std::get<2>(candidates[j]).getHash(data);
		            #endif
	                arma::uvec hashAnswers = hashes.rows(0,groundTruthSets.size()-1);
	                HashIndex ind(hashes,minRetrievalThreshold,maxSearchThreshold,hashBits);

	                //score candidate
	                std::get<1>(candidates[j]) = HashError<HashIndex>::getClassError( hashAnswers,
																		   ind,
																		   groundTruthSets,
																		   maxQuerySize,
																		   minRetrievalThreshold);
		        }

		        //start the boost-loop
		        for (uint64_t i = 0; i < totalHashFunctions; ++i) {


		            //evaluate candidates
		            #pragma omp parallel for
		            for (uint64_t j = 0; j < candidates.size(); ++j) {
		                if (std::get<0>(candidates[j]) >= -9.0) {
		                    std::get<0>(candidates[j]) = HashError<HashIndex>::getFitnessScore(
		                                                    HashError<HashIndex>::getClassScore(
		                                                            groundTruthWeights,
		                                                            fWeight,
		                                                            std::get<1>(candidates[j]),
		                                                            summedTruths)
		                                                    );
		                }
		            }

		            //pick a winner
		            double score = -1.0;
		            uint64_t index = 0;
		            for (uint64_t j = 0; j < candidateHashFunctions; ++j) {
						if (std::get<0>(candidates[j]) > score) {
							score = std::get<0>(candidates[j]);
							index = j;
						}
					}

					//if more hash functions can't improve on the solution, finish early
					if (score == 0.0) break;

					//disable the candidate from future choosing
					std::get<0>(candidates[index]) = -10.0;

					//std::cout << "Candidate for loop " << i << " found with score " << score << std::endl;
		            //add the function to our collection
		            results.addHash(std::get<2>(candidates[index]) );

		            //re-weight the boosting values
		            #pragma omp parallel for
		            for (uint64_t j = 0; j < std::get<1>(candidates[index]).TP.size(); ++j) {
						const auto& p = std::get<1>(candidates[index]).TP[j];
						groundTruthWeights.at(p.first,p.second) = 0.0;
					}
					//we add a normalisation term so that we probabilistically won't upweight things we've zero'd out before
		            double newfWeights = pow(double(std::get<1>(candidates[index]).TN)/totalfWeights,double(i)) * fWeight;


		            //recalculate the sum totals and renormalise
					summedTruths = arma::accu(groundTruthWeights);
					double divisor = summedTruths + newfWeights * totalfWeights;
					fWeight = newfWeights/divisor;
					groundTruthWeights /= divisor;
					summedTruths /= divisor;
		        }



		        return results;
		    }

};



#endif
