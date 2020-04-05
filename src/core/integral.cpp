#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include "../settings/global.h"
#include "integral.h"
#include "tools.h"

using namespace std;
using namespace integral;

// read from file
void integral::read_fcidump(integral::two_body& int2e,
		  	    integral::one_body& int1e,
		  	    double& ecore,
			    string fname,
			    int type){
   auto t0 = global::get_time();
   cout << "\nintegral::read_fcidump fname = " << fname 
	<< " type = " << type << endl;
 
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
	 if(type == 0){
            norb = 2*norb;
	    cout << "norb(spatial) = " << norb/2 << endl;
	 }
         cout << "norb(spinorb) = " << norb << endl;
      }
   }

   // load integrals
   int1e.sorb = norb;
   int1e.init_mem(); 
   int2e.sorb = norb; 
   int2e.init_mem(); 
   cout << "size of int1e (MB) = " << int1e.get_mem() << endl;
   cout << "size of int2e (MB) = " << int2e.get_mem() << endl;

   int i,j,k,l;
   double eri;
   while(!istrm.eof()){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      boost::trim_left(line);
      // might have some problems with taps? 
      boost::split(v,line,boost::is_any_of(" "),boost::token_compress_on);
      eri = stod(v[0]); 
      i = stoi(v[1]); 
      j = stoi(v[2]); 
      k = stoi(v[3]);
      l = stoi(v[4]);
      if(i*j == 0 && k*l == 0){
         ecore = eri;
      }else if(i*j != 0 && k*l == 0){
	 // one-body integral
	 //cout << "i,j,eri=" << i << "," << j << " " << eri << endl;
	 if(type == 0){
	    int ia = 2*i-2, ib = 2*i-1;
	    int ja = 2*j-2, jb = 2*j-1;
	    // expand spatial into spin-orbital integrals
	    //cout << "int1e:" << i << "," << j << endl;
            int1e.set(ia, ja, eri); // AA
            int1e.set(ib, jb, eri); // BB
	 }else if(type == 1){
	    int i1 = i-1, j1 = j-1;
            int1e.set(i1, j1, eri); 
	 }
      }else if(i*j != 0 && k*l != 0){
	 // two-body integral
	 //cout << "i,j,k,l,eri=" << i << "," << j 
	 //     << "," << k << "," << l << " " << eri << endl;
	 if(type == 0){
	    int ia = 2*i-2, ib = 2*i-1;
	    int ja = 2*j-2, jb = 2*j-1;
	    int ka = 2*k-2, kb = 2*k-1;
	    int la = 2*l-2, lb = 2*l-1;
	    // expand spatial into spin-orbital integrals
	    //cout << "int2e:" << i << "," << j << "," << k << "," << l << endl;
            int2e.set(ia, ja, ka, la, eri); // AAAA 
            int2e.set(ib, jb, kb, lb, eri); // BBBB 
            int2e.set(ib, jb, ka, la, eri); // BBAA
	    int2e.set(ia, ja, kb, lb, eri); // AABB 
	    // implement 8-fold symmetric NR integrals 
            int2e.set(ja, ia, ka, la, eri); // AAAA 
            int2e.set(jb, ib, kb, lb, eri); // BBBB 
            int2e.set(jb, ib, ka, la, eri); // BBAA
	    int2e.set(ja, ia, kb, lb, eri); // AABB
	 }else if(type == 1){
	    int i1 = i-1, j1 = j-1, k1 = k-1, l1 = l-1;
            int2e.set(i1, j1, k1, l1, eri); // AAAA 
            int2e.set(j1, i1, k1, l1, eri); // AAAA 
	 }
      }
   }
   istrm.close();

   // get two-index integrals
   int2e.set_JKQ();

   auto t1 = global::get_time();
   cout << "timing for integral::read_fcidump : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   // debug
   //int1e.print();
   //int2e.print();
   //exit(1);
}

// save_text for symmetric 1e integrals
void integral::save_text_sym1e(const std::vector<double>& data,
		               const std::string& fname, 
		               const int prec){
   cout << "\nintegral::save_text_sym1e fname = " << fname << endl; 
   ofstream file(fname+".txt");
   int k = tools::inverse_pair(data.size()).first;
   file << defaultfloat << setprecision(prec); 
   for(int i=0; i<k; i++){
      for(int j=0; j<k; j++){
	 int ij = i>j? i*(i+1)/2+j : j*(j+1)/2+i;
	 file << data[ij] << " ";
      } 
      file << endl;
   }
   file.close();
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
}

// get special parts of integrals 
one_body one_body::get_AA() const{
   one_body int1e(sorb);
   for(int i=0; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [A|A]
      }
   }
   return int1e;
}

one_body one_body::get_BB() const{
   one_body int1e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [B|B]
      }
   }
   return int1e;
}

one_body one_body::get_BA() const{
   one_body int1e(sorb);
   for(int i=0; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
	 int1e.set(i, j, this->get(i,j)); // [A|B],[B|A]
      }
   }
   return int1e;
}

two_body two_body::get_AAAA() const{
   two_body int2e(sorb);
   for(int i=0; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [AA|AA]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

two_body two_body::get_BBBB() const{
   two_body int2e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=1; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|BB]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

two_body two_body::get_BBAA() const{
   two_body int2e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|AA],[AA|BB]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

two_body two_body::get_BAAA() const{
   two_body int2e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=0; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BA|AA],[AB|AA],[AA|BA],[AA|AB]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

two_body two_body::get_BABA() const{
   two_body int2e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=0; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BA|BA],[BA|AB],[AB|BA],[AB|AB]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

two_body two_body::get_BBBA() const{
   two_body int2e(sorb);
   for(int i=1; i<sorb; i+=2){
      for(int j=1; j<sorb; j+=2){
         for(int k=1; k<sorb; k+=2){
            for(int l=0; l<sorb; l+=2){
	       int2e.set(i,j,k,l,this->get(i,j,k,l)); // [BB|BA],[BB|AB],[BA|BB],[AB|BB]
	    }
	 }
      }	 
   }
   int2e.set_JKQ();
   return int2e;
}

// operations
one_body integral::operator +(const one_body& int1eA,
			      const one_body& int1eB){
   one_body int1e(int1eA);
   transform(int1e.data.begin(), int1e.data.end(), int1eB.data.begin(), int1e.data.begin(),
	     [](const double& x, const double& y){ return x+y; });
   return int1e;
}

two_body integral::operator +(const two_body& int2eA,
			      const two_body& int2eB){
   two_body int2e(int2eA);
   transform(int2e.data.begin(), int2e.data.end(), int2eB.data.begin(), int2e.data.begin(),
	     [](const double& x, const double& y){ return x+y; });
   int2e.set_JKQ();
   return int2e;
}
