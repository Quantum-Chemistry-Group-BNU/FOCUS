#include <iostream>
#include <numeric>
#include <fstream>
#include "core/tools.h"
#ifndef SERIAL
#include "core/mpi_wrapper.h"
#endif

void test_print(){
    int nmax = 10000;
    for(int j=0; j<nmax; j++){
        double s = std::log2(double(j+0.1));
        std::cout << " check j=" << j << " s=" << s << std::endl;
    }
}

void test_print2(){
    std::string fname = "test.txt";
    std::ofstream file;
    std::streambuf *psbuf, *backup;
    file.open(fname);
    backup = std::cout.rdbuf(); // back up cout's streambuf
    psbuf = file.rdbuf(); // get file's streambuf
    std::cout.rdbuf(psbuf); // assign streambuf to cout

    test_print();

    std::cout.rdbuf(backup); // restore cout's original streambuf
    file.close();
}

void test_save(const std::vector<double>& data, const int rank, const int data_count){
    // save
    std::ofstream ofs2("data_"+std::to_string(rank)+".data", std::ios::binary);
    ofs2.write(reinterpret_cast<const char*>(data.data()), data_count*sizeof(double));
    ofs2.close();
}

void test_load(std::vector<double>& data, const int rank, const int data_count){
    // save
    std::ifstream ifs2("data_"+std::to_string(rank)+".data", std::ios::binary);
    ifs2.read(reinterpret_cast<char*>(data.data()), data_count*sizeof(double));
    ifs2.close();
}

int main(int argc, char * argv[]) {
    int size = 1, rank = 0;
#ifndef SERIAL
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    rank = world.rank();
    size = world.size();
#endif
    if(rank==0){
        std::cout << "\n### mpisize=" << size << " ###" << std::endl;
    }

    size_t start, end, step;
    start = 1ULL<<20;
    end = 1ULL<<31;
    step = 1ULL<<1;

    for(size_t data_count= start ; data_count< end; data_count *= step)
    {
        if(rank==0)
        {
            std::cout<<std::endl;
            std::cout<<"IO data_size= "
               <<tools::sizeMB<double>(data_count)<<"MB:"
               <<tools::sizeGB<double>(data_count)<<"GB"
               <<std::endl;
        }

        std::vector<double> data(data_count);
        for (size_t i = 0; i < data_count; ++i) {
            data[i] = rank + i;
        }

        auto t0 = tools::get_time();
        test_save(data, rank, data_count);
        auto t1 = tools::get_time();
        test_load(data, rank, data_count);
        auto t2 = tools::get_time();

        double t_save = tools::get_duration(t1-t0);
        double t_load = tools::get_duration(t2-t1);
        std::cout<<" rank=" << rank << " data_count: "<<data_count
            << "  t_save=" << t_save << " speed=" << data_count*sizeof(double)/t_save/std::pow(1024,3) << "GB/S"  
            << "  t_load=" << t_load << " speed=" << data_count*sizeof(double)/t_load/std::pow(1024,3) << "GB/S"
            << std::endl;

#ifndef SERIAL
        world.barrier();
#endif
    }

    return 0;
}
