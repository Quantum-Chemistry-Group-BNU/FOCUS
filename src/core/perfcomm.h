#ifndef SERIAL

#ifndef PERFCOMM_H
#define PERFCOMM_H

#include "tools.h"
#include "mpi_wrapper.h"
#ifdef GPU
#include "gpu/gpu_env.h"
#endif

template <typename Tm>
void perfcomm(const boost::mpi::communicator& world, const size_t data_count){
   const int size = world.size();
   const int rank = world.rank();
   if(rank==0){
      std::cout << "\nperfcomm(broadcast/reduce): size=" << size << ":"
         <<tools::sizeMB<double>(data_count)<<"MB:"
         <<tools::sizeGB<double>(data_count)<<"GB"
         <<std::endl;
   }

   std::vector<Tm> data(data_count, 0);

   world.barrier();
   {
      auto t0 = tools::get_time();
      mpi_wrapper::broadcast(world, data.data(), data_count, 0);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << " mpi broadcast: rank=" << rank << " dt=" << dt
         << " speed=" << tools::sizeGB<Tm>(data_count)/dt << "GB/s" 
         << std::endl;
   }
   world.barrier();
   {
      auto t0 = tools::get_time();
      mpi_wrapper::reduce(world, data.data(), data_count, 0);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << " mpi reduce: rank=" << rank << " dt=" << dt
         << " speed=" << tools::sizeGB<Tm>(data_count)/dt << "GB/s" 
         << std::endl;
   }

#ifdef GPU
#ifdef NCCL
   size_t count = data_count*sizeof(Tm);
   Tm* dev_data = (Tm*)GPUmem.allocate(count);
   if(rank==0) GPUmem.to_gpu(dev_data, data.data(), count);
   cudaDeviceSynchronize();
   world.barrier();
   {
      auto t0 = tools::get_time();
      nccl_comm.broadcast(dev_data, data_count, 0);
      cudaDeviceSynchronize();
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << " nccl broadcast: rank=" << rank << " dt=" << dt
         << " speed=" << tools::sizeGB<Tm>(data_count)/dt << "GB/s" 
         << std::endl;
   }
   world.barrier();
   {
      auto t0 = tools::get_time();
      nccl_comm.reduce(dev_data, data_count, 0);
      cudaDeviceSynchronize();
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << " nccl reduce: rank=" << rank << " dt=" << dt
         << " speed=" << tools::sizeGB<Tm>(data_count)/dt << "GB/s" 
         << std::endl;
   }
   GPUmem.deallocate(dev_data, count);
#endif
#endif
}

#endif

#endif
