#include <iostream>
#include <numeric>
#include <fstream>
#include "core/tools.h"
#ifdef GPU
#include "gpu/gpu_env.h"
#endif
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

#ifdef GPU
   gpu_init(rank);

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

      auto t0x = tools::get_time();
      size_t count = data_count*sizeof(double);
      double* dev_data = (double*)GPUmem.allocate(count);
      auto t1x = tools::get_time();

      if(rank==0) GPUmem.to_gpu(dev_data, data.data(), count);
      cudaDeviceSynchronize();
      world.barrier();

      auto t0y = tools::get_time();
      nccl_comm.broadcast(dev_data, data_count, 0);
      auto t1y = tools::get_time();
      cudaDeviceSynchronize();
      world.barrier();

      auto t0a = tools::get_time();
      GPUmem.to_gpu(dev_data, data.data(), count);
      auto t1a = tools::get_time();
      cudaDeviceSynchronize();
      world.barrier();

      auto t0b = tools::get_time();
#ifdef NCCL
      nccl_comm.reduce(dev_data, data_count, 0);
#endif
      auto t1b = tools::get_time();
      cudaDeviceSynchronize();
      world.barrier();

      auto t0c = tools::get_time();
      GPUmem.to_cpu(data.data(), dev_data, count);
      auto t1c = tools::get_time();
      cudaDeviceSynchronize();
      world.barrier();

      auto t0z = tools::get_time();	
      GPUmem.deallocate(dev_data, count);
      auto t1z = tools::get_time();
      cudaDeviceSynchronize();
      world.barrier();

      double t_cpu2gpu = tools::get_duration(t1a-t0a);
      double t_reduce = tools::get_duration(t1b-t0b);
      double t_gpu2cpu = tools::get_duration(t1c-t0c);
      double t_alloc = tools::get_duration(t1x-t0x);
      double t_bcast = tools::get_duration(t1y-t0y);
      double t_dealloc = tools::get_duration(t1z-t0z);
      std::cout << " rank=" << rank << " data_count: " << data_count
         << "  t_cpu2gpu=" << t_cpu2gpu << " speed=" << count/t_cpu2gpu/std::pow(1024,3) << "GB/s"
         << "  t_reduce="  << t_reduce  << " speed=" << count/t_reduce /std::pow(1024,3) << "GB/s" 
         << "  t_gpu2cpu=" << t_gpu2cpu << " speed=" << count/t_gpu2cpu/std::pow(1024,3) << "GB/s" 
         << "  t_alloc=" << t_alloc << " speed=" << count/t_alloc/std::pow(1024,3) << "GB/s"
         << "  t_bcast="  << t_bcast  << " speed=" << count/t_bcast/std::pow(1024,3) << "GB/s" 
         << "  t_dealloc=" << t_dealloc << " speed=" << count/t_dealloc/std::pow(1024,3) << "GB/s" 
         << std::endl;

   }
#endif

   gpu_finalize();
#endif

   return 0;
}
