#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

string tns::oper_fname(const string scratch, 
  	 	       const comb_coord& p,
		       const string optype){
   string fname = scratch+"/oper_("
	         +to_string(p.first)+","
	         +to_string(p.second)+")_"
		 +optype;
   return fname;
}

/*
void tns::oper_renorm_C(const comb& bra, 
 	                const comb& ket, 
	                comb_coord& coord){
}

void tns::oper_renorm_A(const comb& bra,
		   const comb& ket,
	           comb_coord& coord){
}

void tns::oper_renorm_B(const comb& bra,
		   const comb& ket,
	           comb_coord& coord){
}

void tns::oper_renorm_H(const comb& bra,
		   const comb& ket,
	           comb_coord& coord){
}

void tns::oper_renorm_S(const comb& bra,
		   const comb& ket,
	           comb_coord& coord){
}

void tns::oper_renorm_Q(const comb& bra,
 		   const comb& ket,
	           comb_coord& coord){
}

void tns::oper_renorm_P(const comb_coord& coord,
		  	const comb& bra,
			comst comb& ket){
}
*/

void tns::oper_renorm_C(const comb& bra,
			const comb& ket,
		        const comb_coord& p,
		        const comb_coord& p0,
			const int iop,
			const string scratch){
   cout << "tns::oper_renorm_C iop=" << iop << endl;
   const bool sgnc = true;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   qtensor3 qt3;
   qtensor2 qt2;
   qopers qops;
   string fname = oper_fname(scratch, p, "right_C"); 
   int i = p.first, j = p.second, k = bra.topo[i][j];
   if(iop == 0){
      auto cop = tns::oper_dot_c(); // build
      // previous physical site |r>
      int k0 = bra.rsupport.at(p0)[0];
      for(int s=0; s<2; s++){
         qt3 = contract_qt3_qt2_r(ksite,cop[s]);
         qt2 = contract_qt3_qt3_cr(bsite,qt3,sgnc);
	 qt2.msym = cop[s].msym;
         qops[2*k0+s] = qt2;
      }
      // current physical site |c>
      for(int s=0; s<2; s++){
         qt3 = contract_qt3_qt2_c(ksite,cop[s]); 
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop[s].msym;
         qops[2*k+s] = qt2;
      }
   }else if(iop == 1){
      auto cop = tns::oper_dot_c(); // build
      // previous physical site |r>
      qopers rqops;
      string fname0 = oper_fname(scratch, p0, "right_C"); 
      oper_load(fname0, rqops);
      for(const auto& pr : rqops){
         const auto& rop = pr.second;
         qt3 = contract_qt3_qt2_r(ksite,rop);
         qt2 = contract_qt3_qt3_cr(bsite,qt3,sgnc);
	 qt2.msym = rop.msym;
	 cout << pr.first << endl;
         qops[pr.first] = qt2;
      }
      // current physical site |c>
      for(int s=0; s<2; s++){
         qt3 = contract_qt3_qt2_c(ksite,cop[s]); 
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop[s].msym;
         qops[2*k+s] = qt2;
      }
   }else if(iop == 2){
      auto cop = tns::oper_dot_c(); // load
      auto rop = tns::oper_dot_c(); // build
      exit(1);
   }else if(iop == 3){
      auto rop = tns::oper_dot_c(); // load
      auto cop = tns::oper_dot_c(); // load
      exit(1);
   }else{ 
      cout << "error: no such option for iop=" << iop << endl;
      exit(1);      
   }
   oper_save(fname, qops);
}

void tns::oper_renorm_right(const comb& bra,
			    const comb& ket,
		            const comb_coord& p,
		            const comb_coord& p0,
			    const string scratch){
   cout << "\ntns::oper_renorm_right" << endl;
   int i = p.first, j = p.second;
   int i0 = p0.first, j0 = p0.second; 
   int tp = bra.type.at(p);
   int tp0 = bra.type.at(p0);
   cout << "p=(" << i << "," << j << ")[" << bra.topo[i][j] << "] "
	<< "p0=(" << i0 << "," << j0 << ")[" << bra.topo[i0][j0] << "] " 
	<< "type=[" << tp << "," << tp0 << "]" << endl;
   auto kind = make_pair(tp,tp0);
   if(kind == make_pair(1,0) || kind == make_pair(2,0)){
      oper_renorm_C(bra,ket,p,p0,0,scratch);
      oper_renorm_CA(bra,ket,p,p0,0,scratch);

   }else if(kind == make_pair(1,1) || kind == make_pair(1,3) ||
	    kind == make_pair(0,1) || kind == make_pair(0,3) ||
            kind == make_pair(2,2)){
      oper_renorm_C(bra,ket,p,p0,1,scratch);
      oper_renorm_CA(bra,ket,p,p0,0,scratch);

   }else if(kind == make_pair(3,0)){
      oper_renorm_C(bra,ket,p,p0,2,scratch);
      oper_renorm_CA(bra,ket,p,p0,0,scratch);

   }else if(kind == make_pair(3,1) || kind == make_pair(3,3)){
      oper_renorm_C(bra,ket,p,p0,3,scratch);
      oper_renorm_CA(bra,ket,p,p0,0,scratch);

   }else{
      cout << "error: no such case! (tp,tp0)=" << tp << "," << tp0 << endl;
      exit(1);
   }
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   int nbackbone = bra.nbackbone;
   // loop over internal nodes
   for(int i=nbackbone-2; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
	 auto p0 = make_pair(i+1,0);    
	 oper_renorm_right(bra,ket,p,p0,scratch);
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-2; j>=1; j--){
	    auto pj = make_pair(i,j);
	    auto p0 = make_pair(i,j+1);    
	    oper_renorm_right(bra,ket,pj,p0,scratch);
	 } // j
	 auto p0 = make_pair(i+1,0);
	 oper_renorm_right(bra,ket,p,p0,scratch);
      }else{
	 cout << "error: tp=" << tp << endl;
	 exit(1);
      }
   } // i
}
