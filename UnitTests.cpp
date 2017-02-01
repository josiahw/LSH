#define ARMA_64BIT_WORD
#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <list>
#include <sys/time.h>

//hash functions
#include "RandomProjectionHashFunction.h"
#include "RandomSubSamplingHashFunction.h"
#include "ShiftInvariantKernelHashFunction.h"
#include "DHHashFunction.h"
#include "RandomRotationHashFunction.h"
#include "SparseSignConsistentHashFunction.h"

//data transformers
#include "NullTransformer.h"
#include "MeanTransformer.h"
#include "ThresholdTransformer.h"
#include "PCATransformer.h"
#include "DHHashTransformer.h"

//hash indices
#include "StdHashIndex.h"
#include "ResizeableHashIndex.h"
#include "LSHForestHashIndex.h"
#include "LSHForestHashIndexAsync.h"

//hash collection
#include "HashCollection.h"

//hash collection constructors
#include "RandomConstructor.h"
#include "RDHF.h"
#include "ARDHF.h"
#include "Boost.h"

//unit test class
#include "TestLSH.h"


#include <iostream>


int main (void) {
	const size_t datasize = 100000;
	const size_t querysize = 500;
	const size_t datadim = 128;
	arma::mat data(datadim, datasize, arma::fill::randu);
	arma::mat dataqueries(datadim, querysize, arma::fill::randu);

	size_t hashFuncs = 60;
	size_t hashBits = 16;
	size_t maxQuerySize = 10000;
	size_t maxSearchThreshold = 3;
	size_t minRetrievalThreshold = maxQuerySize;
	size_t maxRetrievalThreshold = maxQuerySize;
	size_t knn = 100;

	//Section 1: test hash indices
	std::cout << "Testing Random Projection Hashing with STDHashIndex" << std::endl;
	auto results = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											StdHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   StdHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   StdHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	auto results2 = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											StdHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   StdHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   StdHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);

	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;


	std::cout << "Testing Random Projection Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);

	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	maxSearchThreshold = 8;
	std::cout << "Testing Random Projection Hashing with LSHForest" << std::endl;
	results = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											LSHForestHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   LSHForestHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   LSHForestHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											LSHForestHashIndex,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   LSHForestHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   LSHForestHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);

	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	maxSearchThreshold = 10;
	std::cout << "Testing Random Projection Hashing with LSHForest_Async" << std::endl;
	results = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											LSHForestHashIndexAsync,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   LSHForestHashIndexAsync,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   LSHForestHashIndexAsync,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<MeanTransformer,
											RandomProjectionHashFunction,
											LSHForestHashIndexAsync,
											HashCollection<MeanTransformer,
														   RandomProjectionHashFunction,
														   LSHForestHashIndexAsync,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   RandomProjectionHashFunction,
						   LSHForestHashIndexAsync,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);

	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

















	//Section 2: test hash functions
	maxSearchThreshold = 3;
	std::cout << "Testing Random Subsampled Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<ThresholdTransformer,
											RandomSubSamplingHashFunction,
											ResizeableHashIndex,
											HashCollection<ThresholdTransformer,
														   RandomSubSamplingHashFunction,
														   StdHashIndex,
														   arma::mat>,
											arma::mat>,
						   ThresholdTransformer,
						   RandomSubSamplingHashFunction,
						   StdHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<ThresholdTransformer,
											RandomSubSamplingHashFunction,
											ResizeableHashIndex,
											HashCollection<ThresholdTransformer,
														   RandomSubSamplingHashFunction,
														   StdHashIndex,
														   arma::mat>,
											arma::mat>,
						   ThresholdTransformer,
						   RandomSubSamplingHashFunction,
						   StdHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	std::cout << "Testing Shift Invariant Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<MeanTransformer,
											ShiftInvariantKernelHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   ShiftInvariantKernelHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   ShiftInvariantKernelHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<MeanTransformer,
											ShiftInvariantKernelHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   ShiftInvariantKernelHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   ShiftInvariantKernelHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	std::cout << "Testing Random Rotation Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<PCATransformer,
											RandomRotationHashFunction,
											ResizeableHashIndex,
											HashCollection<PCATransformer,
														   RandomRotationHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   PCATransformer,
						   RandomRotationHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<PCATransformer,
											RandomRotationHashFunction,
											ResizeableHashIndex,
											HashCollection<PCATransformer,
														   RandomRotationHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   PCATransformer,
						   RandomRotationHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	std::cout << "Testing Double Hadamard Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<DHHashTransformer,
											DoubleHadamardHashFunction,
											ResizeableHashIndex,
											HashCollection<DHHashTransformer,
														   DoubleHadamardHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   DHHashTransformer,
						   DoubleHadamardHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<DHHashTransformer,
											DoubleHadamardHashFunction,
											ResizeableHashIndex,
											HashCollection<DHHashTransformer,
														   DoubleHadamardHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   DHHashTransformer,
						   DoubleHadamardHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;

	std::cout << "Testing Sparse Sign Consistent Hashing with Robin Hood Hash Index" << std::endl;
	results = TestLSH<RandomConstructor<MeanTransformer,
											SparseSignConsistentHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   SparseSignConsistentHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   SparseSignConsistentHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHTimings(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	results2 = TestLSH<RandomConstructor<MeanTransformer,
											SparseSignConsistentHashFunction,
											ResizeableHashIndex,
											HashCollection<MeanTransformer,
														   SparseSignConsistentHashFunction,
														   ResizeableHashIndex,
														   arma::mat>,
											arma::mat>,
						   MeanTransformer,
						   SparseSignConsistentHashFunction,
						   ResizeableHashIndex,
						   arma::mat
				   			>::TestLSHAccuracy(data,
				   					 dataqueries,
				   					 arma::Mat<uint32_t>(), //groundtruth not used for random
				   					 datadim,
				   					 datasize,
				   					 querysize,
				   					 hashFuncs,
				   					 hashBits,
				   					 hashFuncs, //candidate hash functs not used for random
				   					 maxSearchThreshold,
				   					 maxQuerySize,
									 maxRetrievalThreshold,
				   					 minRetrievalThreshold,
				   					 knn
				   					);
	std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
	std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
	std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
	std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
	std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;
	return 0;
};
