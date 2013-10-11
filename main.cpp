#include <iostream>
#include <armadillo>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "LSH.h"
#include <stdlib.h>
//using namespace std;

static void show_usage(std::string name)
{
    std::cout << "Usage: " << name << " <option(s)> SOURCES" << std::endl
              << "Options:" << std::endl
              << "\t-h,--help\t\tShow this help message" << std::endl
              << "\t-if,--if   INPUTFILE\tSpecify the input file" << std::endl
              << "\t-of,--of   OUTPUTFILE\tSpecify the output file" << std::endl
              << "\t-t,--tables NUMTABLES\tSpecify the number of hashtables" << std::endl
              << "\t-b,--bits  BITSPERHASH\tSpecify thenumber of bits per hash" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc > 8) { //if there are too many/too few arguments, die
        show_usage(argv[0]);
        return 1;
    }

    //pre-set some variables to defaults
    uint32_t tables = 30;
    uint32_t bits = 30;
    std::string infile = "/home/merlz/Desktop/mnist_test.csv";
    std::string outfile = "/home/merlz/Desktop/mnistnn_32tables_32bits.csv";

    //set all command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if ((arg == "-if") || (arg == "--if")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                infile = argv[i++]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cout << "-if option requires one argument." << std::endl;
                return 1;
            }
        } else if ((arg == "-of") || (arg == "--of")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                outfile = argv[i++]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "-of option requires one argument." << std::endl;
                return 1;
            }
        } else if ((arg == "-t") || (arg == "--tables")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                tables = atoi(argv[i++]); // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "--tables option requires one argument." << std::endl;
                return 1;
            }
        } else if ((arg == "-b") || (arg == "--bits")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                bits = atoi(argv[i++]); // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--bits option requires one argument." << std::endl;
                return 1;
            }
        } else {
            show_usage(argv[0]);
            return 1;
        }
    }
    arma::mat data;
    data.load(infile,arma::csv_ascii);
    std::cout << "Data loaded, " << data.n_cols << " " << data.n_rows << std::endl;
    LSH<float> table(tables,bits);
    auto start = std::chrono::steady_clock::now();
    table.loadDataSet(data);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Table built in " << double((end-start).count())/double(std::chrono::steady_clock::period::den) << "s" << std::endl;
    double totalvecs = 0.;
    std::vector<std::vector<size_t>> allNeighbours;
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < data.n_rows; ++i) { //
        allNeighbours.push_back(table.queryKNN_Euclidean(data.row(i)));
        //allNeighbours.push_back(table.query(data.row(i)));
        totalvecs += allNeighbours.back().size();
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Queries done in "<< double((end-start).count())/double(std::chrono::steady_clock::period::den) <<"s, avg: " << totalvecs/double(data.n_rows) << std::endl;

    //output to a file so we can compare
    std::ofstream out(outfile);
    for (auto& n : allNeighbours) {
        out << n[0];
        for (size_t i = 1; i < n.size(); ++i) {
            out << "," << n[i];
        }
        out << std::endl;
    }
    out.close();
    return 0;
}
