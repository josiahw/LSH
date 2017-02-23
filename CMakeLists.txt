# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

# Project's name

project(SearchLSH)
# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O3 -flto -march=native -mtune=native")




#find libraries
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#check C++11 support:
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -ggdb")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/hashFunction")
add_subdirectory(hashFunction)
include_directories("${PROJECT_SOURCE_DIR}/hashIndex")
add_subdirectory(hashIndex)
include_directories("${PROJECT_SOURCE_DIR}/testing")
add_subdirectory(testing)
include_directories("${PROJECT_SOURCE_DIR}/hashBitOptimizers")
add_subdirectory(hashBitOptimizers)

#create executables
add_executable(UnitTests ${PROJECT_SOURCE_DIR}/UnitTests.cpp)
add_executable(ThesisCh2Tests ${PROJECT_SOURCE_DIR}/ThesisCh2Tests.cpp)

target_link_libraries(UnitTests
                    testlsh
                    hashcollection
                    randomconstructor
                    rdhf
                    ardhf
                    boost
                    lshforesthashindex
                    lshforesthashindexasync
                    resizeablehashindex
                    stdhashindex
                    randomprojectionhashfunction
                    shiftinvariantkernelhashfunction
                    randomrotationhashfunction
                    randomsubsamplinghashfunction
                    meantransformer
                    pcatransformer
                    thresholdtransformer
                    sparsesignconsistenthashfunction
                    ${ARMADILLO_LIBRARIES})

target_link_libraries(ThesisCh2Tests
                    testlsh
                    hashcollection
                    randomconstructor
                    rdhf
                    ardhf
                    boost
                    lshforesthashindex
                    lshforesthashindexasync
                    resizeablehashindex
                    stdhashindex
                    randomprojectionhashfunction
                    shiftinvariantkernelhashfunction
                    randomrotationhashfunction
                    randomsubsamplinghashfunction
                    meantransformer
                    pcatransformer
                    thresholdtransformer
                    sparsesignconsistenthashfunction
                    ${ARMADILLO_LIBRARIES})
