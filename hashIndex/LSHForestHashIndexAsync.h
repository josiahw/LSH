//#include "_HashIndex.h"
#ifndef LSHFORESTHASHINDEXASYNC_H
#define LSHFORESTHASHINDEXASYNC_H

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
#include <deque>

#include <iostream>
#include "LSHForestSupport.h"

class LSHForestHashIndexAsync {
	private:
	    uint64_t minRetrieved,
	             nBits,
	             qDepth,
	             cntr,
	             rootNode;


		std::vector<HashNode> hashTree;

	    int32_t insertElement(const uint64_t& hashCode,
	                           const int32_t& node,
	                           const uint64_t& length = 0) {
	        int32_t newNode = node;
	        //we need to create a new node
	        if (node == -1) {
	            newNode = hashTree.size();
	            hashTree.push_back({hashCode,
                                    64, //uint32_t(length),
                                    -1,
                                    -1,
                                    //arma::uvec(1)
                                    std::vector<uint32_t>(1,cntr)
                                    });
	            //hashTree.back().children[0] = cntr;
	            cntr += 1;

	        //we've hit a child node with no branches
	        } else if (hashTree[node].children.size() > 0) {

	            if (hashCode == hashTree[node].prefix) {
	                hashTree[node].children.resize(hashTree[node].children.size() + 1);
	                hashTree[node].children[hashTree[node].children.size()-1] = cntr;
	            } else {
	                newNode = hashTree.size();

	                //find new prefix length
	                uint32_t newLength = length;
	                while (newLength < 64) {
	                    if ( (hashCode >> (64 - newLength)) !=
	                         (hashTree[node].prefix >> (64 - newLength))) {
	                        break;
	                    }
	                    ++newLength;
	                }
	                --newLength; //go back to the level where both are the same

	                //add a new child node
	                hashTree.push_back({hashCode,
                                        64,
                                        -1,
                                        -1,
                                        //arma::uvec(1)
                                        std::vector<uint32_t>(1,cntr)
                                        });
	                //hashTree.back().children[0] = cntr;

	                //reset the length on the other terminal node
	                hashTree[node].length = newLength+1;

	                if (hashCode > hashTree[node].prefix) {
	                    hashTree.push_back({hashCode >> (64 - newLength),
                                            newLength,
                                            newNode,
                                            node,
                                            //arma::uvec()
                                            std::vector<uint32_t>(0)
                                            });
	                } else {
	                    hashTree.push_back({hashCode >> (64 - newLength),
                                            newLength,
                                            node,
                                            newNode,
                                            std::vector<uint32_t>(0)
                                            });
	                }
	                ++newNode;
	            }
	            ++cntr;

	        //we've hit a decision node and need to insert a node above it.
	        } else if (hashTree[node].prefix != hashCode >> (64 - hashTree[node].length)) {
	            newNode = hashTree.size();

                //find new prefix length
                uint32_t newLength = length;
                uint64_t hc = hashTree[node].prefix << (64 - hashTree[node].length); //this makes it a pseudohashcode
                while (newLength < hashTree[node].length) {
                    if ( (hashCode >> (64 - newLength)) !=
                         (hc >> (64 - newLength))) {
                        break;
                    }
                    ++newLength;
                }
                --newLength; //go back to the level where both are the same
                //add a new child node
                hashTree.push_back({hashCode,
                                    64, //newLength+1,
                                    -1,
                                    -1,
                                    //arma::uvec(1)
                                    std::vector<uint32_t>(1,cntr)
                                    });
                //ashTree.back().children[0] = cntr;


                //XXX: needs help
                if ( (hashCode >> (63 - newLength)) > (hc >> (63 - newLength))) {
                    hashTree.push_back({hashCode >> (64 - newLength),
                                        newLength,
                                        newNode,
                                        node,
                                        std::vector<uint32_t>(0)
                                        });
                } else {
                    hashTree.push_back({hashCode >> (64 - newLength),
                                        newLength,
                                        node,
                                        newNode,
                                        std::vector<uint32_t>(0)
                                        });
                }
                ++newNode;
	            ++cntr;

	        //we've hit a decision node and should descend
	        } else { //if (hashTree[node].prefix == hashCode >> (64 - hashTree[node].length)) {
	            uint64_t l = hashTree[node].length+1;
	            int32_t n;
	            if ( (hashCode >> (63 - hashTree[node].length)) & 1 ) {
	                n = hashTree[node].leftChild;
	                hashTree[node].leftChild = insertElement(hashCode,
	                                                         n,
	                                                         l);
	            } else {
	                n = hashTree[node].rightChild;
	                hashTree[node].rightChild = insertElement(hashCode,
                                                              n,
                                                              l);
	            }

	        }

	        return newNode;
	    };

	    //collect all values in a subtree
	    void collect(const int32_t& node, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
            if (uniqueResults.size() == results.n_elem) {
                return;
            }
	        if (hashTree[node].children.size() > 0) {
	            for (const auto& c : hashTree[node].children) {
	                if (not uniqueResults.count(c)) {
						results[uniqueResults.size()] = c;
						uniqueResults.insert(c);
	                    if (uniqueResults.size() >= results.n_elem) {
	                        return;
	                    }
	                }
	            }
	        } else {
	            collect(hashTree[node].leftChild,results,uniqueResults);
	            collect(hashTree[node].rightChild,results,uniqueResults);
	        }
	    }

	    //collect all values in a subtree
	    template<class SET = std::set<uint32_t>>
	    void collect(const int32_t& node, SET& uniqueResults) {
            if (uniqueResults.size() >= minRetrieved) {
                return;
            }

	        if (hashTree[node].children.size() > 0) {
	            for (const auto& c : hashTree[node].children) {
	                if (not uniqueResults.count(c)) {
						uniqueResults.insert(c);

	                    /*if (uniqueResults.size() >= minRetrieved) {
	                        return;
	                    }*/
	                }
	            }
	        } else {
	            collect(hashTree[node].leftChild,uniqueResults);
	            collect(hashTree[node].rightChild,uniqueResults);
	        }
	    }

	public:

		LSHForestHashIndexAsync() {
		    ;
		    };

		LSHForestHashIndexAsync(const arma::uvec& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {
		    minRetrieved = queryThreshold;
		    qDepth = queryDepth;
		    nBits = 64-bitSize;
		    cntr = 0;
	        //maxDist = 0;

		    minRetrieved = queryThreshold;

		    hashTree.reserve(std::min<uint64_t>(2*hashCodes.n_elem,2ull<<bitSize));
		    rootNode =  -1;
		    for (const auto& h : hashCodes) {
		        rootNode = insertElement(h,rootNode,nBits);
		    }
		}

		LSHForestHashIndexAsync(const uint64_t& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    minRetrieved = queryThreshold;
		    qDepth = queryDepth;
		    nBits = 64-bitSize;
		    cntr = 0;
	        //maxDist = 0;

		    minRetrieved = queryThreshold;

		    rootNode = -1;
		    hashTree.reserve(std::min<uint64_t>(2*hashCodes,2ull<<bitSize));

		}

		//chunked function for inserting elements
		void countIndices(const arma::uvec& hashCodes) {
		    for (const auto& h : hashCodes) {
		        rootNode = insertElement(h,rootNode,nBits);
		    }
		}

		void fillQuery(const uint64_t& hashCode, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
		    //1: descend the tree to the closest leaf, saving nodes as we go
		    int32_t counter = 0;
		    std::vector<int32_t> nodevals(64);
		    nodevals[0] = rootNode;
		    while (hashTree[nodevals[counter]].children.size() == 0) {
		        const auto& hc = hashTree[nodevals[counter]];
		        ++counter;
		        if ( hashCode >> (63 - hc.length) & 1) {
		            nodevals[counter] = hc.leftChild;
		        } else {
		            nodevals[counter] = hc.rightChild;
		        }

		    }

		    //2: add immediate neighbours
		    collect(nodevals[counter], results, uniqueResults);

		    //3: ascend up the tree, adding alternate branches until we hit the return requirement
		    const uint64_t minRet = uniqueResults.size()+minRetrieved;
		    while (counter > 0
		           and minRet > uniqueResults.size()
		           and 63-hashTree[nodevals[counter]].length < qDepth) {
		        --counter;
		        (hashTree[nodevals[counter]].leftChild == nodevals[counter+1]) ?
                    collect(hashTree[nodevals[counter]].rightChild, results, uniqueResults) :
                    collect(hashTree[nodevals[counter]].leftChild, results, uniqueResults);
		    }
		}

		template<class SET = std::set<uint32_t>>
		void fillQuerySet(const uint64_t& hashCode, SET& uniqueResults) {

		    //1: descend the tree to the closest leaf, saving nodes as we go
		    int32_t counter = 0;
		    std::vector<int32_t> nodevals(64);
		    nodevals[0] = rootNode;
		    while (hashTree[nodevals[counter]].children.size() == 0) {
		        const auto& hc = hashTree[nodevals[counter]];
		        ++counter;
		        if ( hashCode >> (63 - hc.length) & 1) {
		            nodevals[counter] = hc.leftChild;
		        } else {
		            nodevals[counter] = hc.rightChild;
		        }
		    }

		    //2: add immediate neighbours
		    collect(nodevals[counter], uniqueResults);

		    //3: ascend up the tree, adding alternate branches until we hit the return requirement
		    const uint64_t minRet = uniqueResults.size()+minRetrieved;
		    while (counter > 0
		           and minRet > uniqueResults.size()
		           and 63-hashTree[nodevals[counter]].length < qDepth) {
		        --counter;
		        (hashTree[nodevals[counter]].leftChild == nodevals[counter+1]) ?
	                    collect(hashTree[nodevals[counter]].rightChild, uniqueResults) :
	                    collect(hashTree[nodevals[counter]].leftChild, uniqueResults);
		    }
		}

		uint64_t size() const {
			return cntr;
		}

		/* XXX: I spose we should implement this sometime.
		void saveTable(std::string fileName) {
		    ;
		}*/
};

#endif
