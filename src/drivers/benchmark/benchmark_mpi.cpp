#include <iostream>
#include <numeric>
#include <fstream>
#include "core/tools.h"
#ifndef SERIAL
#include "core/mpi_wrapper.h"
#endif

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

#ifndef SERIAL        
   for(size_t data_count= start ; data_count< end; data_count *= step)
   {
      if(rank==0)
      {
         std::cout<<std::endl;
         std::cout<<"reduce data_size= "
            <<tools::sizeMB<double>(data_count)<<"MB:"
            <<tools::sizeGB<double>(data_count)<<"GB"
            <<std::endl;
      }

      std::vector<double> data(data_count);
      for (size_t i = 0; i < data_count; ++i) {
         data[i] = rank + i;
      }

      auto t0 = tools::get_time();
      world.barrier();
      mpi_wrapper::reduce(world, data.data(), data_count, 0);
      world.barrier();
      auto t1 = tools::get_time();
      double t_reduce = tools::get_duration(t1-t0);
      std::cout<<" rank=" << rank << " data_count: "<<data_count
         << "  t_reduce=" << t_reduce << " speed=" << data_count*sizeof(double)/t_reduce/std::pow(1024,3) << "GB/s"  
         << std::endl;

      world.barrier();
   }
#endif

   return 0;
}
