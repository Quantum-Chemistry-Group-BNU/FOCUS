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
      std::cout << "CPUmem: VM:" << vm_usage/1024.0 << "GB"   // Virtual Memory used
         << " RSS:" << resident_set/1024.0 << "GB";   // Actual Memory used

      // Timing
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1.e6; // user time used
      double sys  = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1.e6; // sys time used
      std::cout << "Timing: User:" << user << "S" 
         << " Sys:" << sys << "S"
         << " Total:" << user+sys << "S"
         << std::endl;
   }

} // ctns

#endif
