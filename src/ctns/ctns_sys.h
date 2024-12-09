#ifndef CTNS_SYS_H
#define CTNS_SYS_H

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
//
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <cstdlib>

#include <iostream>
#include <cstdlib>

#ifdef __linux__
#include <fstream>
#include <sstream>
#include <unistd.h>

inline double getMemoryUsageLinux() {
    std::ifstream file("/proc/self/statm");
    long totalPages, residentPages;

    if (file) {
        file >> totalPages >> residentPages;
        long pageSize = sysconf(_SC_PAGE_SIZE);  // Get the system's page size
        return residentPages * pageSize / std::pow(1024.0, 3);  // Convert to GB
    }

    return 0;
}
#endif

#ifdef __APPLE__
#include <malloc/malloc.h>
#include <sys/sysctl.h>

#include <mach/mach.h>

inline double getMemoryUsageMac() {
    // Get current memory usage using malloc_info on macOS (works with debugging)
    // Alternatively, using malloc_zone_statistics can give detailed memory usage
    malloc_statistics_t stats;
    malloc_zone_statistics(malloc_default_zone(), &stats);
    std::cout << "lzd:" << stats.size_in_use << std::endl;
    return stats.size_in_use / std::pow(1024.0, 3); // Return used size in GB
}

#endif

inline double getMemoryUsage() {
#ifdef __linux__
    return getMemoryUsageLinux();
#elif defined(__APPLE__)
    return getMemoryUsageMac();
#else
    std::cerr << "warning in getMemoryUsage: Unsupported platform!" << std::endl;
    return 0;
#endif
}

namespace ctns{

   extern size_t totalAllocatedMemory;

   // attempts to read the system-dependent data for a process' virtual memory
   // size and resident set size, and return the results in KB.
   inline void get_sys_status(){
      using std::ios_base;
      using std::ifstream;
      using std::string;

      // 'file' stat seems to give the most reliable results
      ifstream stat_stream("/proc/self/stat",ios_base::in);

      // dummy vars for leading entries in stat that we don't care about
      string pid, comm, state, ppid, pgrp, session, tty_nr;
      string tpgid, flags, minflt, cminflt, majflt, cmajflt;
      string utime, stime, cutime, cstime, priority, Nice;
      string O, itrealvalue, starttime;

      // the two fields we want
      unsigned long vsize;
      long rss;

      stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
         >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
         >> utime >> stime >> cutime >> cstime >> priority >> Nice
         >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

      stat_stream.close();

      long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
      double vm_usage     = vsize /(1024.0*1024.0);
      double resident_set = rss * page_size_kb /(1024.0);
      std::cout << std::scientific;
      std::cout << "CPUmem: VM:" << vm_usage/1024.0 << "GB"   // Virtual Memory used
         << " RSS:" << resident_set/1024.0 << "GB";   // Actual Memory used

      // Timing
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1.e6; // user time used
      double sys  = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1.e6; // sys time used
      std::cout << "  Timing: User:" << user << "S" 
         << " Sys:" << sys << "S"
         << " Total:" << user+sys << "S"
         << std::endl;


      std::ifstream file("/proc/self/status");
      std::string line;
      long memoryUsage = 0;

      while (std::getline(file, line)) {
         if (line.find("VmRSS") != std::string::npos) {  // Resident Set Size (RAM)
            size_t pos = line.find_last_of(" \t");
            memoryUsage = std::stol(line.substr(pos + 1));  // Get the value (in KB)
            break;
         }
      }
      std::cout << "memoryUsage=" << memoryUsage/1024.0 << "GB" << std::endl;

      std::cout << "usage=" << usage.ru_maxrss/1024.0 << "GB" << std::endl;  // Memory in kilobytes

      std::ifstream file2("/proc/self/statm");
      long totalPages, residentPages;

      if (file2) {
         file >> totalPages >> residentPages;
         long pageSize = sysconf(_SC_PAGE_SIZE);  // Get the system's page size
         std::cout << "lzd memory usage (RSS in KB):"
           << residentPages * pageSize / 1024
           << " KB" << std::endl;
      }

      std::cout << "Current memory usage: " << getMemoryUsage() << " GB" << std::endl;
      std::cout << totalAllocatedMemory << ":"
         << totalAllocatedMemory/std::pow(1024.0,3) << "GB"
         << std::endl;
   }

} // ctns

#endif
