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

namespace ctns{

// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
inline void get_sys_status(const std::string msg=""){
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
   double vm_usage     = vsize /(1024.0*1024.0*1024.0);
   double resident_set = rss * page_size_kb /(1024.0*1024.0);

   // Timing
   struct rusage usage;
   getrusage(RUSAGE_SELF, &usage);
   double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1.e6; // user time used
   double sys  = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1.e6; // sys time used

   // Available mem
   std::ifstream meminfo("/proc/meminfo");
   std::string token, key = "MemAvailable";
   size_t value, memrest;
   while(meminfo >> token){
      if(token == key + ":"){
         if(meminfo >> value){
            memrest = value/(1024.0*1024);  
         }
      }
   }
 
   std::cout << msg
             << "  VM: " << vm_usage << " GB" 	     // Virtual Memory used
             << "  RSS: " << resident_set << " GB"   // Actual Memory used
             << " -->" << memrest  
             << "  User: " << user << " s" //  用户空间使用的时间
             << "  Sys: " << sys << " s"   //  内核空间使用的时间
             << "  Total: " << user+sys << " s" // 总共使用的时钟
             << std::endl;
}

} // ctns

#endif
