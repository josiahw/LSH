#ifndef HASHERROR_H
#define HASHERROR_H
#define ARMA_DONT_USE_WRAPPER
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

struct classificationErrors {
    std::vector<std::pair<uint64_t,uint64_t>> TP;
    std::vector<std::pair<uint64_t,uint64_t>> FN;
    uint64_t TN;
    uint64_t FP;
};

template<class HashIndex>
class HashError {
	private:

	public:

		static classificationErrors getClassError(const arma::uvec& queryHashes,
		                                          HashIndex& dataHashes,
		                                          const std::vector<std::set<uint64_t>>& groundTruthSets,
		                                          const uint64_t& maxQuerySize = 2000,
		                                          const uint64_t& minQuerySize = 200) {

		    //for each query
		    uint64_t cnt = 0;
		    std::vector<std::pair<uint64_t,uint64_t>> truePositives;
		    truePositives.reserve(queryHashes.n_elem * groundTruthSets[0].size());
		    std::vector<std::pair<uint64_t,uint64_t>> falseNegatives;
		    uint64_t trueNegatives = 0, falsePositives = 0;
			std::set<uint32_t> uniqueResults;
			//arma::Col<uint32_t> results(maxQuerySize);
		    for (const auto& q : queryHashes) {
				const auto& gcSet = groundTruthSets[cnt];

				//query the hash table.
				uniqueResults.clear();
				#ifndef LSHFOREST
				dataHashes.fillQuerySet(q,uniqueResults);
				#else
				std::vector<int32_t> nvals;
				int cnt2 = 0;
				while (uniqueResults.size() < minQuerySize and cnt2 < dataHashes.qDepth) {
		            dataHashes.fillQuerySet(q,uniqueResults,nvals,cnt2,minQuerySize);
		            ++cnt2;
		        }
		        #endif

		        //build a list of true positives
		        double tp0 = truePositives.size();
		        const auto& start = gcSet.begin();
		        if (uniqueResults.size() < gcSet.size()) {
					for (const auto& h : uniqueResults) { //use the ordered constraint on groundtruth to index into the weights matrix
						if (gcSet.count(h)) {
							truePositives.emplace_back(cnt,std::distance(start,gcSet.find(h)));
						}
					}
				} else {
					for (const auto& h : gcSet) { //use the ordered constraint on groundtruth to index into the weights matrix
						if (uniqueResults.count(h)) {
							truePositives.emplace_back(cnt,std::distance(start,gcSet.find(h)));
						}
					}
				}

		        //count tn and fp
		        falsePositives += uniqueResults.size() - (truePositives.size() - tp0);
		        trueNegatives += dataHashes.size() - uniqueResults.size() - ( gcSet.size() - (truePositives.size() - tp0) );
		        ++cnt;
		    }

		    return {truePositives,falseNegatives,trueNegatives,falsePositives};
		}

		static arma::vec getClassScore(const arma::mat& classificationWeights,
		                               const double& falseWeight,
		                               const classificationErrors& classifications,
		                               const double& psumTotal = -1.0) {
		    double tpTotal = 0.0;
		    for (const auto& q : classifications.TP) {
		        tpTotal += classificationWeights.at(q.first,q.second);
		    }

		    double fnTotal;
		    if (psumTotal < 0.0) {
		        fnTotal =  0.0;
		        for (const auto& q : classifications.FN) {
		            fnTotal += classificationWeights.at(q.first,q.second);
		        }
		    } else {
		        fnTotal = psumTotal - tpTotal;
		    }

		    const double fpTotal = classifications.FP*falseWeight;
		    const double tnTotal = classifications.TN*falseWeight;

		    return arma::vec4{tpTotal,fnTotal,tnTotal,fpTotal};
		}

		static double getFitnessScore(const arma::vec4& classScores) {
		    const double recall = classScores[0]/(classScores[0]+classScores[1]);
		    const double precision = classScores[0]/(classScores[0]+classScores[3]);
		    return recall*precision*precision;
		}
};



#endif
