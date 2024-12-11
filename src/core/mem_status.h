#ifndef MEM_STATUS_H
#define MEM_STATUS_H

#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef __linux__
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <malloc.h>  // For malloc_stats

double getTotalMemory() {
   std::ifstream meminfo("/proc/meminfo");
   std::string line;
   long totalMemory = 0;
   // Read the /proc/meminfo file and find "MemTotal"
   while (std::getline(meminfo, line)) {
      if (line.find("MemTotal") != std::string::npos) {
         sscanf(line.c_str(), "MemTotal: %ld kB", &totalMemory);
         break;
      }
   }
   return totalMemory / std::pow(1024.0,2);  // Convert to GB
}

double getAvailableMemory() {
   std::ifstream meminfo("/proc/meminfo");
   std::string line;
   long availableMemory = 0;
   while (std::getline(meminfo, line)) {
      if (line.find("MemAvailable") != std::string::npos) {
         // The value is in kB, so we return it in GB
         sscanf(line.c_str(), "MemAvailable: %ld kB", &availableMemory);
         return availableMemory / std::pow(1024.0,2);  // Convert to GB
      }
   }
   return 0;  // Return 0 if the value couldn't be found
}
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>

// Function to get Total Memory in bytes
double getTotalMemory() {
   uint64_t totalMemory = 0;
   size_t length = sizeof(totalMemory);
   // sysctl to get the total physical memory (HW_MEMSIZE)
   if (sysctlbyname("hw.memsize", &totalMemory, &length, NULL, 0) == 0) {
      return totalMemory / std::pow(1024.0,3);
   } else {
      std::cerr << "Failed to get total memory" << std::endl;
      return 0;  // Return -1 if error occurs
   }
}

double getAvailableMemory() {
   vm_statistics_data_t vmStats;
   mach_msg_type_number_t vmStatsCount = HOST_VM_INFO_COUNT;
   long availableMemory = 0;
   // Get virtual memory statistics (including free and inactive pages)
   if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmStats, &vmStatsCount) == KERN_SUCCESS) {
      // Sum of free pages and inactive pages (macOS can reclaim inactive pages quickly)
      availableMemory = (vmStats.free_count + vmStats.inactive_count) * sysconf(_SC_PAGESIZE); // in bytes
      return availableMemory / std::pow(1024.0,3);
   } else {
      std::cerr << "Failed to get available memory" << std::endl;
      return 0;
   }
}
#endif

#ifdef TCMALLOC
#include <tcmalloc.h>
#include <malloc_extension.h>

void release_freecpumem(){
   // Released free memory to the system
   MallocExtension::instance()->ReleaseFreeMemory();
}
#endif

void get_cpumem_status(const int rank, const int level=0, const std::string msg=""){
#ifdef TCMALLOC
   static double previous = 0.0;
   size_t total_allocated;
   size_t heap_size;
   size_t pageheap_free_bytes;
   MallocExtension::instance()->GetNumericProperty("generic.current_allocated_bytes", &total_allocated);
   MallocExtension::instance()->GetNumericProperty("generic.heap_size", &heap_size);
   MallocExtension::instance()->GetNumericProperty("tcmalloc.pageheap_free_bytes", &pageheap_free_bytes);
   double current = total_allocated / std::pow(1024.0, 3);
#endif
   std::cout << "rank=" << rank
      << " CPUmem(GB):"
      << std::scientific << std::setprecision(3)
#ifdef TCMALLOC
      << " used=" << current
      << " pageheap_free=" << (pageheap_free_bytes/std::pow(1024.0,3))
      << " change=" << (current-previous)
      << " heap=" << (heap_size/std::pow(1024.0,3))
#endif
      << " avail=" << getAvailableMemory()
      << " total=" << getTotalMemory()
      << " " << msg
      << std::endl;
#ifdef TCMALLOC
   previous = current;
   if(level > 0){
      // Create a buffer to store stats
      char buffer[2048];  // Make sure the buffer is large enough for the stats
      int buffer_length = sizeof(buffer);
      MallocExtension::instance()->GetStats(buffer, buffer_length);
      std::cout << "memory usage stats:\n" << buffer << std::endl;
   }
#endif
}

#ifdef SERIAL

void mem_check(const bool ifgpu){
   double avail_cpu = getAvailableMemory();
   double total_cpu = getTotalMemory();
   std::cout << std::scientific << std::setprecision(3);
   std::cout << "mem_check CPUmem(GB): rank=0"
	  << " avail=" << avail_cpu 
	  << " total=" << total_cpu 
	  << std::endl;
#ifdef GPU
   if(ifgpu){
      size_t avail, total;
      CUDA_CHECK(cudaMemGetInfo(&avail, &total));
      double avail_gpu = avail / std::pow(1024.0,3);
      double total_gpu = total / std::pow(1024.0,3);
      std::cout << "mem_check GPUmem(GB): rank=0"
	     << " avail=" << avail_gpu 
	     << " total=" << total_gpu 
	     << std::endl;
   }
#endif 
}

#else

#include "perfcomm.h"

void mem_check(const bool ifgpu, const boost::mpi::communicator& world){
   int size = world.size();
   int rank = world.rank();
   double avail_cpu = getAvailableMemory();
   double total_cpu = getTotalMemory();
   std::vector<double> avail_cpus, total_cpus;
   boost::mpi::gather(world, avail_cpu, avail_cpus, 0);
   boost::mpi::gather(world, total_cpu, total_cpus, 0);
   if(rank == 0){
      std::cout << std::scientific << std::setprecision(3);
      for(int i=0; i<size; i++){
         std::cout << "memcheck CPUmem(GB): rank=" << i
		<< " avail=" << avail_cpus[i]
	       	<< " total=" << total_cpus[i]
	       	<< std::endl;
      }
      auto ptr = std::minmax_element(avail_cpus.begin(), avail_cpus.end());
      double diff = (*ptr.second-*ptr.first);
      std::cout << "memcheck CPUmem(GB): min=" << *ptr.first
	      << " max=" << *ptr.second
	      << " diff=" << diff
	      << std::endl;
      if(diff > 5.0) std::cout << "WARNING: diff(CPUmem) is greater than 5GB!" << std::endl;
   }
#ifdef GPU
   if(ifgpu){
      size_t avail, total;
      CUDA_CHECK(cudaMemGetInfo(&avail, &total));
      double avail_gpu = avail / std::pow(1024.0,3);
      double total_gpu = total / std::pow(1024.0,3);
      std::vector<double> avail_gpus, total_gpus;
      boost::mpi::gather(world, avail_gpu, avail_gpus, 0);
      boost::mpi::gather(world, total_gpu, total_gpus, 0);
      if(rank == 0){
         for(int i=0; i<size; i++){
            std::cout << "memcheck GPUmem(GB): rank=" << i
           	<< " avail=" << avail_gpus[i]
                << " total=" << total_gpus[i]
                << std::endl;
         }
         auto ptr = std::minmax_element(avail_gpus.begin(), avail_gpus.end());
	 double diff = (*ptr.second-*ptr.first);
         std::cout << "memcheck GPUmem(GB): min=" << *ptr.first
   	      << " max=" << *ptr.second
   	      << " diff=" << diff 
   	      << std::endl;
	 if(diff > 2.0) std::cout << "WARNING: diff(GPUmem) is greater than 2GB!" << std::endl;
      }
   }
#endif 
}

#endif

#endif
