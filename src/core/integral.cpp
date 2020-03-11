#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "tools.h"
#include "integral.h"
#include "../settings/global.h"

using namespace std;
using namespace integral;

// read from file
void integral::read_fcidump(integral::two_body& int2e,
		  	    integral::one_body& int1e,
		  	    double& ecore,
			    string fname){
   auto t0 = global::get_time();
   cout << "\nintegral::read_fcidump" << endl;
   cout << "fname = " << fname << endl;
 
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }

   // parse FCIDUMP file
   int icounter = 0;
   int norb = 0;
   vector<string> v;
   string line;
   while(!istrm.eof() && icounter < 4){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      icounter++;
      if(icounter == 1){
	 boost::trim_left(line); // in case there is a space in FCIDUMP
	 boost::split(v,line,boost::is_any_of(" ,="),boost::token_compress_on);
         norb = stoi(v[2]);
         cout << "norb(spatial) = " << norb << endl;
         cout << "norb(spinorb) = " << 2*norb << endl;
      }
   }

   // load integrals
   int1e.sorb = 2*norb; 
   int2e.sorb = 2*norb; 
   int i,j,k,l;
   double eri;
   while(!istrm.eof()){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      boost::trim_left(line);
      boost::split(v,line,boost::is_any_of(" "),boost::token_compress_on);
      eri = stod(v[0]); 
      i = stoi(v[1]); 
      j = stoi(v[2]); 
      k = stoi(v[3]);
      l = stoi(v[4]);
      if(i*j == 0 && k*l == 0){
         ecore = eri;
      }else if(i*j != 0 && k*l == 0){
	 auto ia = 2*i-2, ib = 2*i-1;
	 auto ja = 2*j-2, jb = 2*j-1;
	 // expand spatial into spin-orbital integrals
         int1e.set(ia, ja, eri); // AA
         int1e.set(ib, jb, eri); // BB
      }else if(i*j != 0 && k*l != 0){
	 auto ia = 2*i-2, ib = 2*i-1;
	 auto ja = 2*j-2, jb = 2*j-1;
	 auto ka = 2*k-2, kb = 2*k-1;
	 auto la = 2*l-2, lb = 2*l-1;
         int2e.set(ia, ja, ka, la, eri); // AAAA 
         int2e.set(ib, jb, kb, lb, eri); // BBBB 
         int2e.set(ib, jb, ka, la, eri); // BBAA
	 int2e.set(ia, ja, kb, lb, eri); // AABB 
	 // implement 8-fold symmetric NR integrals 
         int2e.set(ja, ia,  ka, la, eri); // AAAA 
         int2e.set(jb, ib,  kb, lb, eri); // BBBB 
         int2e.set(jb, ib,  ka, la, eri); // BBAA
	 int2e.set(ja, ia,  kb, lb, eri); // AABB
      }
   }
   istrm.close();

   cout << "size of int1e (MB) = " << int1e.get_mem_space() << endl;
   cout << "size of int2e (MB) = " << int2e.get_mem_space() << endl;
   auto t1 = global::get_time();
   cout << "timing for integral::read_fcidump: " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;

   // debug
   //int1e.print();
   //int2e.print();
   //exit(1);
}

// print for debug
void one_body::print(){
   cout << "\none_body::print" << endl;
   cout << setprecision(12);
   for(int i=0; i<sorb; i++){
      for(int j=0; j<sorb; j++){
	 cout << "i,j=" << i << "," << j << " val=" << this->get(i,j) << endl;
      }
   }
   cout << "unordered_map:" << endl;
   for(auto dt : data){
      auto p = tools::inverse_pair(dt.first);
      cout << "addr=" << dt.first << " i,j="
	   << p.first << "," << p.second 
	   << " eri=" << dt.second << endl;
   }
}

void two_body::print(){
   cout << "\ntwo_body::print" << endl;
   cout << setprecision(12);
   for(int i=0; i<sorb; i++){
      for(int j=0; j<sorb; j++){
	 for(int k=0; k<sorb; k++){
 	    for(int l=0; l<sorb; l++){
	       cout << "i,j,k,l=" 
	            << i << "," << j << "," << k << "," << l 
	            << " -> " << i/2+1 << "," << j/2+1 << "," << k/2+1 << "," << l/2+1
	            << " : " << i%2 << "," << j%2 << "," << k%2 << "," << l%2  
		    << " val=" << this->get(i,j,k,l) << endl;
	    }
	 }
      }
   }
   cout << "unordered_map:" << endl;
   for(auto dt : data){
      auto q = inverse_quad(dt.first);
      size_t i = std::get<0>(q); 
      size_t j = std::get<1>(q); 
      size_t k = std::get<2>(q); 
      size_t l = std::get<3>(q); 
      cout << "addr=" << dt.first << " i,j,k,l=" 
	   << i << "," << j << "," << k << "," << l 
	   << " -> " << i/2+1 << "," << j/2+1 << "," << k/2+1 << "," << l/2+1
	   << " : " << i%2 << "," << j%2 << "," << k%2 << "," << l%2  
	   << " eri=" << dt.second << endl;
   }
}

// get special parts of integrals 
one_body one_body::get_AA() const{
   one_body int1e;
   int1e.sorb = sorb;
   for(int i=0; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [A|A]
      }
   }
   return int1e;
}

one_body one_body::get_BB() const{
   one_body int1e;
   int1e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [B|B]
      }
   }
   return int1e;
}

one_body one_body::get_BA() const{
   one_body int1e;
   int1e.sorb = sorb;
   for(int i=0; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [A|B],[B|A]
      }
   }
   return int1e;
}

two_body two_body::get_AAAA() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=0; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [AA|AA]
	    }
	 }
      }	 
   }
   return int2e;
}

two_body two_body::get_BBBB() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=1; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|BB]
	    }
	 }
      }	 
   }
   return int2e;
}

two_body two_body::get_BBAA() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|AA],[AA|BB]
	    }
	 }
      }	 
   }
   return int2e;
}

two_body two_body::get_BAAA() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BA|AA],[AB|AA],[AA|BA],[AA|AB]
	    }
	 }
      }	 
   }
   return int2e;
}

two_body two_body::get_BABA() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BA|BA],[BA|AB],[AB|BA],[AB|AB]
	    }
	 }
      }	 
   }
   return int2e;
}

two_body two_body::get_BBBA() const{
   two_body int2e;
   int2e.sorb = sorb;
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|BA],[BB|AB],[BA|BB],[AB|BB]
	    }
	 }
      }	 
   }
   return int2e;
}

// operations
one_body integral::operator +(const one_body& int1eA,
			      const one_body& int1eB){
   one_body int1e(int1eA);
   for(const auto& p : int1eB.data){
      auto search = int1e.data.find(p.first);
      if(search == int1e.data.end()) int1e.data[p.first] = 0.0;
      int1e.data[p.first] += int1eB.data.at(p.first);
   }
   return int1e;
}

two_body integral::operator +(const two_body& int2eA,
			      const two_body& int2eB){
   two_body int2e(int2eA);
   for(const auto& p : int2eB.data){
      auto search = int2e.data.find(p.first);
      if(search == int2e.data.end()) int2e.data[p.first] = 0.0;
      int2e.data[p.first] += int2eB.data.at(p.first);
   }
   return int2e;
}
