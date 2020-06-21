#ifndef FCI_UTIL_H
#define FCI_UTIL_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <complex>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"

namespace fci{

// type information
template <typename Tm>
inline bool is_complex(){ return false; }
template <>
inline bool is_complex<std::complex<double>>(){ return true; }

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space,
		      const int istart=0);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dimA0 = 0, dimB0 = 0, dimA, dimB;
};

// compute coupling of states:
// basically describe how two states are differ defined by diff_type,
// which partition the cartesian space {(I,J)} into disjoint subspace!
struct coupling_table{
   public:
      void get_Cmn(const fock::onspace& space, 
		   const bool Htype,
		   const int istart=0);
   public:
      std::vector<std::set<int>> C11; // differ by single (sorted, binary_search)
      std::vector<std::set<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::set<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
};

// linked list - store each row H[i] as a list
template <typename Tm>
struct sparse_hamiltonian{
   public:
      void get_hamiltonian(const fock::onspace& space,
		           const product_space& pspace,
		           const coupling_table& ctabA,
			   const coupling_table& ctabB,
			   const integral::two_body<Tm>& int2e,
			   const integral::one_body<Tm>& int1e,
			   const double ecore,
		   	   const bool Htype,
			   const int istart=0){
         bool debug = true;
         auto t0 = tools::get_time();
         cout << "\nsparse_hamiltonian::get_hamiltonian" 
              << " dim0 = " << istart << " dim = " << space.size() << endl; 
         // initialization for the first use
         if(istart == 0){
            diag.clear();
            connect.clear();
            value.clear();
            diff.clear();
         }
         // diagonal 
         dim = space.size();
         diag.resize(dim);
         for(size_t i=istart; i<dim; i++){
            diag[i] = fock::get_Hii(space[i],int2e,int1e) + ecore;
         }
         auto ta = tools::get_time();
         if(debug) cout << "timing for diagonal : " << setprecision(2) 
              	        << tools::get_duration(ta-t0) << " s" << endl;
         // off-diagonal 
         connect.resize(dim);
         value.resize(dim);
         diff.resize(dim);
	 get_Haaaa(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
         get_Hbbbb(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 get_Habba(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 /*
         if(Htype){
	    get_Haaab(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
            get_Hbbba(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	    get_Haabb(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 }
	 */
   auto t1 = tools::get_time();
   cout << "timing for sparse_hamiltonian::get_hamiltonian : " 
	<< setprecision(2) << tools::get_duration(t1-t0) << " s" << endl;
      }

      // (C11+C22)_A*C00_B:
      // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single/double
      // 	 	       {I_B,J_B} differ by zero (I_B=J_B)
      void get_Haaaa(const fock::onspace& space,
                     const product_space& pspace,
                     const coupling_table& ctabA,
                     const coupling_table& ctabB,
                     const integral::two_body<Tm>& int2e,
                     const integral::one_body<Tm>& int1e,
                     const int istart,
		     const bool debug){
         auto t0 = tools::get_time();
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
               for(const auto& pja : pspace.colB[ib]){
                  int ja = pja.first;
                  int j = pja.second;
                  if(j >= i) continue; 
                  // check connectivity <I_A|H|J_A>
                  auto pr = pspace.spaceA[ia].diff_type(pspace.spaceA[ja]);
                  if(pr == make_pair(1,1)){
                     auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
                     connect[i].push_back(j);
                     value[i].push_back(pr.first);
                     diff[i].push_back(pr.second);
                  }else if(pr == make_pair(2,2)){
                     auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
                     connect[i].push_back(j);
                     value[i].push_back(pr.first);
                     diff[i].push_back(pr.second);
                  }
               } // ja
            } // ib
         } // ia
         auto t1 = tools::get_time();
         if(debug) cout << "timing for (C11+C22)_A*C00_B : " << setprecision(2) 
              	        << tools::get_duration(t1-t0) << " s" << endl;
      }

      // C00_A*(C11+C22)_B:
      // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by zero (I_A=J_A)
      // 			    {I_B,J_B} differ by single/double
      void get_Hbbbb(const fock::onspace& space,
                     const product_space& pspace,
                     const coupling_table& ctabA,
                     const coupling_table& ctabB,
                     const integral::two_body<Tm>& int2e,
                     const integral::one_body<Tm>& int1e,
                     const int istart,
		     const bool debug){
         auto t0 = tools::get_time();
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
      	 int ib = pib.first;
      	 int i = pib.second;
      	 if(i < istart) continue; // incremental build
               for(const auto& pjb : pspace.rowA[ia]){
      	    int jb = pjb.first;
      	    int j = pjb.second;
      	    if(j >= i) continue; 
      	    // check connectivity <I_B|H|J_B>
      	    auto pr = pspace.spaceB[ib].diff_type(pspace.spaceB[jb]);
      	    if(pr == make_pair(1,1)){
      	       auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
      	       connect[i].push_back(j);
      	       value[i].push_back(pr.first);
      	       diff[i].push_back(pr.second);
      	    }else if(pr == make_pair(2,2)){
      	       auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
      	       connect[i].push_back(j);
      	       value[i].push_back(pr.first);
      	       diff[i].push_back(pr.second);
      	    }
      	 } // jb
            } // ib
         } // ia
         auto t1 = tools::get_time();
         if(debug) cout << "timing for C00_A*(C11+C22)_B : " << setprecision(2) 
      		  << tools::get_duration(t1-t0) << " s" << endl;
      }
      
   // 3. C11_A*C11_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single
   // 			    {I_B,J_B} differ by single
       void get_Habba(const fock::onspace& space,
                     const product_space& pspace,
                     const coupling_table& ctabA,
                     const coupling_table& ctabB,
                     const integral::two_body<Tm>& int2e,
                     const integral::one_body<Tm>& int1e,
                     const int istart,
		     const bool debug){
         auto t0 = tools::get_time();
    for(int ia=0; ia<pspace.dimA; ia++){
      for(const auto& pib : pspace.rowA[ia]){
	 int ib = pib.first;
	 int i = pib.second;
	 if(i < istart) continue; // incremental build
	 for(int ja : ctabA.C11[ia]){
	    for(const auto& pjb : pspace.rowA[ja]){
	       int jb = pjb.first;
	       int j = pjb.second;	       
   	       if(j >=i) continue;
	       auto search = ctabB.C11[ib].find(jb);
	       if(search != ctabB.C11[ib].end()){
	          auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
	          connect[i].push_back(j);
	          value[i].push_back(pr.first);
	          diff[i].push_back(pr.second);
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
   auto t1 = tools::get_time();
   if(debug) cout << "timing for C11_A*C11_B : " << setprecision(2) 
   		  << tools::get_duration(t1-t0) << " s" << endl;
       }

      // compare with full construction
      void check(const fock::onspace& space,
	 	 const integral::two_body<Tm>& int2e,
		 const integral::one_body<Tm>& int1e,
		 const double ecore,
		 const double thresh=1.e-10){
   	 cout << "\nsparse_hamiltonian::check" << endl;
         int dim = connect.size();
	 linalg::matrix<Tm> H1(dim,dim);
         for(int i=0; i<dim; i++){
            for(int jdx=0; jdx<connect[i].size(); jdx++){
               int j = connect[i][jdx];
               H1(i,j) = value[i][jdx];
	       H1(j,i) = tools::conjugate(value[i][jdx]); 
            }
	    H1(i,i) += diag[i];
         }
         // compared againts construction by Slater-Condon rule
         auto H2 = fock::get_Ham(space,int2e,int1e,ecore);
         for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
               if(abs(H1(i,j))<1.e-8 && abs(H2(i,j))<1.e-8) continue;
               if(abs(H1(i,j)-H2(i,j))<1.e-8) continue;
               cout << "i,j=" << i << "," << j << " "
                    << space[i] << " " << space[j]  
                    << " val=" << H1(i,j) << "," << H2(i,j) 
                    << " diff=" << H1(i,j)-H2(i,j) 
                    << " num=" << space[i].diff_num(space[j]) 
                    << endl;
            }
         } 
	 double diff = normF(H2-H1);
         cout << "|H2-H1|=" << diff << endl;
	 if(diff > thresh){
	    cout << "error: difference is greater than thresh=" << thresh << endl; 
	 }
      }

      // analyze the magnitude of Hij
      void analysis(){
   	 cout << "\nsparse_hamiltonian::analysis" << endl;
         map<int,int,greater<int>> bucket;
         double size = 1.e-20; // avoid divide zero in the special case H=0;
         double Hsum = 0;
         for(int i=0; i<dim; i++){
            size += connect[i].size();
            for(int jdx=0; jdx<connect[i].size(); jdx++){
      	       int j = connect[i][jdx];
      	       double aval = abs(value[i][jdx]);
      	       if(aval > 1.e-8){
      	          int n = floor(log10(aval));
      	          bucket[n] += 1;
      	          Hsum += aval;
      	       }
            }
         }
         double avc = 2.0*size/dim;
         cout << "dim = " << dim
         	<< "  avc = " << defaultfloat << fixed << avc
         	<< "  per = " << defaultfloat << setprecision(3) << avc/(dim-1)*100 << endl; 
         cout << "average size |Hij| = " << scientific << setprecision(1) << Hsum/size << endl;
         // print statistics by magnitude 
         double accum = 0.0;
         for(const auto& pr : bucket){
            double per = pr.second/size*100;
            int n = pr.first;
            accum += per;
            cout << "|Hij| in 10^" << showpos << n+1 << "-10^" << n << " : " 
      	         << defaultfloat << noshowpos << fixed << setw(8) << setprecision(3) << per << " " 
      	         << defaultfloat << noshowpos << fixed << setw(8) << setprecision(3) << accum 
      	         << endl;
         }
      }
   public:
      int dim;
      // diagonal part: H[i,i]
      std::vector<double> diag;   
      // lower-riangular part: H[i,j] (i>j)
      std::vector<std::vector<int>> connect; // connected by H
      std::vector<std::vector<Tm>> value; // H[i][j] 
      std::vector<std::vector<long>> diff; // packed orbital difference 
};

} // fci

#endif
