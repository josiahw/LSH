#ifndef HASHINDEX_H
#define HASHINDEX_H

#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <unordered_set>
#include <list>

#include <iostream>



std::vector<uint64_t> queryDist(uint64_t maxDist, uint64_t bitSize) {
    std::vector<uint64_t> results(0);
    if (maxDist == 0) {
        results.push_back(0);
    } else if (maxDist == 1) {
        for (uint64_t i = 0; i < bitSize; ++i) {
            results.push_back(1ull << i);
        }
    } else {
        for (uint64_t i = maxDist-1; i < bitSize; ++i) {
            const auto answers = queryDist(maxDist-1,i);
            for (const auto& a : answers) {
                results.push_back(a + (1ull << i ));
            }
        }
    }
    return results;
};

std::vector<std::vector<uint64_t>> getBallDistances(uint64_t maxDist, uint64_t bitSize) {
    std::vector<std::vector<uint64_t>> dists(maxDist);
    for (uint64_t i = 0; i < maxDist; ++i) {
        dists[i] = queryDist(i,bitSize);
        std::sort (dists[i].begin(), dists[i].end());
        if (i > 0) {
            dists[0].insert(dists[0].end(),dists[i].begin(),dists[i].end());
        }
    }
    dists.resize(1);
    return dists;
};

class HashIndex {
	private:
		arma::Col<uint32_t> indices, binAddresses, bInd;
		#ifdef SHERWOORD
		sherwood_map<uint64_t,uint32_t> binLocations;
		#else
		std::unordered_map<uint64_t,uint32_t> binLocations;
		#endif
		uint64_t minRetrieved, cntr;
		std::vector<std::vector<uint64_t>> binDists;

	public:

		HashIndex() {
		    binLocations.reserve(800000);
		    };

		HashIndex(const arma::uvec& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;

		    //count up all bins
		    binLocations.reserve(hashCodes.n_elem);
		    binLocations.reserve(2*hashCodes.n_elem);
		    binLocations.max_load_factor(0.5);
		    for (const auto& h : hashCodes) {
		        #ifdef SHERWOORD
		        auto& b = binLocations.find(h)
		        if (b == binLocations.end()) {
		            binLocations[h] = 0;
		        } else {
		            ++(b->second);
		        }
		        #else
		        ++binLocations[h];
		        #endif
		    }

		    //set the bin addresses
		    cntr = 0;
		    uint64_t total = 0;
		    binAddresses.resize(binLocations.size() + 1);

		    for (auto& b : binLocations) {
		        binAddresses[cntr] = total;
		        total += b.second;
		        b.second = cntr;
		        ++cntr;
		    }
		    binAddresses[binLocations.size()] = total;
		    //std::cout << binLocations.size() << std::endl;
		    //insert all the indices
		    arma::Col<uint32_t> bInd = binAddresses;
		    indices.resize(hashCodes.n_elem);
		    cntr = 0;
		    for (const auto& h : hashCodes) {
		        const auto& bl = binLocations[h];
		        indices[bInd[bl]] = cntr;
		        ++bInd[binLocations[bl]];
		        ++cntr;
		    }
		}

		HashIndex(const uint64_t& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;
		    //binLocations.reserve(2*hashCodes);
		    //binLocations.max_load_factor(0.5);
		    indices.resize(hashCodes);

		}

		//chunked Function 1: countIndices
		void countIndices(const arma::uvec& hashCodes) {
		    //count up all bins
		    for (const auto& h : hashCodes) {
		        #ifdef SHERWOORD
		        auto& b = binLocations.find(h)
		        if (b == binLocations.end()) {
		            binLocations[h] = 0;
		        } else {
		            ++(b->second);
		        }
		        #else
		        ++binLocations[h];
		        #endif
		    }
		}

		//function2: prepareIndices
		void prepareIndices() {
		    //set the bin addresses
		    cntr = 0;
		    uint64_t total = 0;
		    binAddresses.resize(binLocations.size() + 1);
		    for (auto& b : binLocations) {
		        binAddresses[cntr] = total;
		        total += b.second;
		        b.second = cntr;
		        ++cntr;
		    }
		    binAddresses[binLocations.size()] = total;

		    //insert all the indices
		    bInd = binAddresses;
		    cntr = 0;
		}


		    //function3: loadHashes
		void loadHashes(const arma::uvec& hashCodes) {
		    for (const auto& h : hashCodes) {
		        const auto& bl = binLocations[h];
		        indices[bInd[bl]] = cntr;
		        ++bInd[bl];
		        ++cntr;
		    }
		}



		void fillQuery(const uint64_t& hashCode, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
		    //this fills an external query - so we can have a hash collection outside it collecting queries

		    //expand the hash-ball until we meet our minimum query threshold
		    uint64_t cnt = 0;
		    //for (const auto& b : binDists) {
		        for (const uint64_t& v : binDists.front()) {
		            const auto hc = hashCode^v;
					const auto& binLoc = binLocations.find(hc);
					if (binLoc != binLocations.end()) {
						const uint32_t addr = binLoc->second;
						const uint32_t end = binAddresses[addr+1];
						for ( uint32_t i = binAddresses[addr]; i < end; ++i) {
							const uint32_t& index = indices[i];
							if (not uniqueResults.count(index)) {
								results[uniqueResults.size()] = index;
								uniqueResults.insert(index);
								++cnt;
								if (uniqueResults.size() >= results.n_elem) break;
							}
						}
						if (cnt > minRetrieved or uniqueResults.size() >= results.n_elem) break;

					}
		        }
				//if (cnt > minRetrieved or uniqueResults.size() >= results.n_elem) break;
		    //}

		}

		template<class SET = std::set<uint32_t>>
		void fillQuerySet(const uint64_t& hashCode, SET& uniqueResults) {
		    //this fills an external query set, making some locality assumptions to help the boosting process.
		    const uint64_t minRet = uniqueResults.size()+minRetrieved;
		    //expand the hash-ball until we meet our minimum query threshold
		    //for (const auto& b : binDists) {
		        for (const uint64_t& v : binDists.front()) {
		            const auto hc = hashCode^v;
					const auto& binLoc = binLocations.find(hc);
					if (binLoc != binLocations.end()) {
						const uint32_t& addr = binLoc->second;
						const uint32_t& end = binAddresses[addr+1];
						//uniqueResults.insert(indices.begin_row(binAddresses[addr]),
						//                     indices.end_row(end-1));
						for ( uint32_t i = binAddresses[addr]; i < end and uniqueResults.size() < minRet; ++i) {
							uniqueResults.insert(indices[i]);
						}
						if (uniqueResults.size() >= minRet) break;

					}
		        }
				//if (uniqueResults.size() >= minRet) break;
		    //}

		}

		uint64_t size() const {
			return indices.n_elem;
		}

		/* XXX: I spose we should implement this sometime.
		void saveTable(std::string fileName) {
		    ;
		}*/
};

class HashCollection {
	private:
		PCATransformer downsampler;
		std::vector<HashFunction> hashes;
		std::vector<HashIndex> indices;
		uint64_t querySize, minQuery, hashBits;
	public:
		uint64_t queryDepth;
		HashCollection(const arma::mat& data, const uint64_t& nbits, const uint64_t& querysize, const uint64_t& minquery, const uint64_t& querydepth, const uint64_t& bits) {
			//make the pca transformer
			downsampler = PCATransformer(data.rows(0,std::min<uint64_t>(30000-1,data.n_rows-1)),nbits);
			querySize = querysize;
			minQuery = minquery;
			queryDepth = querydepth;
			hashBits = bits;
		}

		void addHash(const HashFunction& h) {
			hashes.push_back(h);
		}

		void buildDB(const DataSet<float_t>& data) {
			//XXX: this assumes we can grab the data in one chunk
			auto tf = downsampler.transform(data.getRaw());
			indices.clear();
			indices.resize(hashes.size());
			//build the hash databases
			#pragma omp parallel for
			for (uint64_t i = 0; i < hashes.size(); ++i) {
				indices[i] = HashIndex(hashes[i].getHash(tf),minQuery,queryDepth,hashBits);
			}
		}

		void buildChunkedDB(const DataSet<float_t>& data, const uint64_t& chunkSize = 500000) {
			indices.clear();
			indices.resize(hashes.size());

			//init the hashes
			#pragma omp parallel for
			for (uint64_t i = 0; i < hashes.size(); ++i) {
				indices[i] = HashIndex(data.length,minQuery,queryDepth,hashBits);
			}

			//count bucket sizes
			for (uint64_t i = 0; i < data.length; i += chunkSize) {
				auto tf = downsampler.transform(data.getUntransformed(i,std::min<uint64_t>(i+chunkSize-1,data.length-1)));
				//build the hash databases
				#pragma omp parallel for
				for (uint64_t j = 0; j < hashes.size(); ++j) {
					indices[j].countIndices(hashes[j].getHash(tf));
				}
			}

			//calculate bucket locations
			#pragma omp parallel for
			for (uint64_t i = 0; i < hashes.size(); ++i) {
				indices[i].prepareIndices();
			}

			//insert all data
			for (uint64_t i = 0; i < data.length; i += chunkSize) {
				auto tf = downsampler.transform(data.getUntransformed(i,std::min<uint64_t>(i+chunkSize-1,data.length-1)));
				//build the hash databases
				#pragma omp parallel for
				for (uint64_t j = 0; j < hashes.size(); ++j) {
					indices[j].loadHashes(hashes[j].getHash(tf));
				}
			}

		}

		std::pair<arma::Mat<uint32_t>,arma::Mat<float>> batchQuery(const arma::Mat<float>& q, const uint64_t& kneighbours, const DataSet<float>& baseData) {
			//this does the transform, hash, and query for a single item
			auto tf = downsampler.transform(q);
			/*arma::Mat<uint32_t> resultinds(q.n_cols,kneighbours);
			arma::Mat<float> resultdists(q.n_cols,kneighbours);
			resultinds.fill(0);
			resultdists.fill(0.0);*/
			arma::umat queryHashes(q.n_cols,indices.size());
			for (uint64_t i = 0; i < indices.size(); ++i) {
			    queryHashes.col(i) = hashes[i].getHash(tf);
			}
			tf.reset();

			std::unordered_set<uint32_t> uniqueResults;
			uniqueResults.reserve(querySize);
			//std::set<uint32_t> uniqueResults;
			#pragma omp parallel for private(uniqueResults)
			for (uint64_t j = 0; j < q.n_cols; ++j) {
			    //std::set<uint32_t> uniqueResults;
			    for (uint64_t i = 0; i < indices.size() and uniqueResults.size() <= querySize; ++i) {
				    indices[i].fillQuerySet(queryHashes(j,i), uniqueResults);
			    }
			    arma::Col<uint32_t> results(uniqueResults.size());
			    uint32_t cnt = 0;
			    for (const auto& r : uniqueResults) {
			        results[cnt] = r;
			        ++cnt;
			    }

			    arma::Mat<float> vals = baseData.getRaw(results);
			    vals.each_col() -= q.col(j);
			    arma::Col<float> dists = arma::sum(arma::square(vals),0).t();
			    auto cmp = ([&dists](const uint32_t& a, const uint32_t& b) {
							return dists[a]<dists[b];
						});

			    std::vector<uint32_t> distanceStack;
			    distanceStack.reserve(kneighbours);
			    cnt = std::min<uint32_t>(kneighbours,results.n_elem);
			    for (uint32_t i = 0; i < cnt; ++i) {
			        distanceStack.emplace_back(i);
			    }
			    std::make_heap(distanceStack.begin(),distanceStack.end(),cmp);
			    float current = dists[distanceStack.front()];
			    for (uint32_t i = distanceStack.size(); i < results.n_elem; ++i) {
			        if (dists[i] < current) {
			            std::pop_heap(distanceStack.begin(),distanceStack.end(),cmp);
			            distanceStack[kneighbours-1] = i;
					    std::push_heap(distanceStack.begin(),distanceStack.end(),cmp);
					    current = dists[distanceStack.front()];
					}
			    }

			    if (results.n_elem) {
					//resultinds.row(j).fill(distanceStack.front()); //just check for dist=0 instead of this
					const arma::uvec tmpInds = arma::conv_to<arma::uvec>::from(distanceStack);
					resultinds.submat(j,0,j,distanceStack.size()-1) = results(tmpInds).t();
					resultdists.submat(j,0,j,distanceStack.size()-1) = arma::sqrt(dists(tmpInds).t());
				}

			}
			return std::make_pair(resultinds,resultdists);
		}

		#ifdef RANDOMSUBSAMPLING
		arma::umat transform(const arma::mat& data) {
			//access to the transform so we don't double up when training
			return downsampler.transform(data);
		}
		#else
		arma::mat transform(const arma::mat& data) {
			//access to the transform so we don't double up when training
			return downsampler.transform(data);
		}
		#endif

};

#endif
