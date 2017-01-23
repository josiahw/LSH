import numpy

def dropFirstRow(inFile, outFile, dataWidth, dataLength):
    """
    The BigANN datasets prepend each row with an index term.
    Due to the ordering of the data, this gets in the way when
    using memory mapped files to back large dataset arrays.

    This function removes the prepended indices to improve
    interaction with memory mapping.
    """
    numpy.fromfile(inFile,
                dtype=numpy.float32
                ).reshape((dataLength, dataWidth))[:,1:].tofile(outFile)

dropFirstRow('sift/sift_base.fvecs', 'sift_mmapready', 129, 1000000)
dropFirstRow('sift/sift_query.fvecs', 'sift_queries_mmapready', 129, 10000)
dropFirstRow('gist/gist_base.fvecs', 'gist_mmapready', 961, 1000000)
dropFirstRow('gist/gist_query.fvecs', 'gist_queries_mmapready', 961, 1000)