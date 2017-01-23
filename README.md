# LSH

This repository implements the nearest-neighbour locality-sensitive-hashing code and benchmarks found in my thesis, titled "Improved Similarity Search for Large Data in Machine Learning and Robotics". This code is released under the CC-BY-NC-SA Creative Commons License.

This code implements: a hash index using the C standard library's hash map; a hash index using a custom-written robin-hood-hashing scheme; and both synchronous and asynchronous query LSH forests. The hash functions implemented include: Random projections; Random subsampling; PCA with random rotations; Shift-invariant Kernel hashing; and Double Hadamard Hashing. Several hash collection optimisation algorithms are also included, which will be detailed soon.

## Usage

See the unit tests for examples of how to instantiate an LSH index. The "batchQuery()" function is the intended default interface for external use, as the closest K neighbours found are returned after distance post-processing.

## Reproducing Thesis Results

My thesis makes use of the BigANN datasets heavily for benchmarking, since they are large and high-dimensional. The original page can be found at: http://corpus-texmex.irisa.fr

To download and format the data for reproducing the search tree results in my thesis, run:
* $ ./support/getdatasets.sh
* $ python support/reformatdatasets.py

From the directory that the formatted datasets are contained in, run:
* $ ./bin/ThesisCh2Tests > thesisch2results.csv

Finally, to generate the figures contained in the thesis, run:
* $ python support/fig2_14.py

## Citing

If you use this code in your research, please cite my thesis:

@phdthesis{walker2017similaritysearch,
  title={Improved Similarity Search for Large Data in Machine Learning and Robotics},
  author={Walker, Josiah},
  year={2017},
  school={School of Electrical Engineering and Computer Science},
  publisher = {The University of Newcastle},
  address = {Callaghan, NSW, Australia}
}
