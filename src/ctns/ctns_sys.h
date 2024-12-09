#ifndef CTNS_SYS_H
#define CTNS_SYS_H

#ifdef __linux__
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <malloc.h>  // For malloc_stats

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

inline double getAvailableMemoryLinux() {
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

   inline void get_sys_status(){
      std::cout << "CPUmem(GB):"
        << std::setprecision(3)
        << " used=" << getMemoryUsage() 
        << " avail=" << getAvailableMemoryLinux() 
        << std::endl;
   }

} // ctns

#endif
