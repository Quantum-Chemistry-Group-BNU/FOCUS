#include <iomanip>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <cassert>
#include "input.h"
#include "../core/tools.h"

using namespace std;
using namespace input;

//
// CTNS
//
void params_ctns::read(ifstream& istrm){
   if(debug_input) cout << "params_ctns::read" << endl; 
   run = true;
   // helpers
   vector<params_sweep> tmp_ctrls;
   // read 
   string line;
   while(true){
      line.clear();	   
      getline(istrm,line);
      std::cout << line << std::endl;
      if(line.empty() || line[0]=='#'){
         continue; // skip empty and comments    
      }else if(line.substr(0,4)=="$end"){ 
         break;
      }else if(line.substr(0,5)=="qkind"){
         istringstream is(line.substr(5));
         is >> qkind;
      }else if(line.substr(0,13)=="topology_file"){
         istringstream is(line.substr(13));
         is >> topology_file;
      }else if(line.substr(0,7)=="verbose"){
         verbose = stoi(line.substr(7));
      }else if(line.substr(0,9)=="task_init"){
         task_init = true;
      }else if(line.substr(0,10)=="task_sdiag"){
         task_sdiag = true;
      }else if(line.substr(0,9)=="task_dmrg"){
         task_ham = true;
         task_opt = true;
      }else if(line.substr(0,8)=="task_ham"){
         task_ham = true;
      }else if(line.substr(0,8)=="task_opt"){
         task_opt = true;
      }else if(line.substr(0,11)=="task_oodmrg"){
         task_oodmrg = true;
      }else if(line.substr(0,8)=="task_vmc"){
         task_vmc = true;
      }else if(line.substr(0,11)=="task_expand"){
         task_expand = true;
      }else if(line.substr(0,13)=="task_tononsu2"){
         task_tononsu2 = true;
      }else if(line.substr(0,12)=="task_schmidt"){
         task_schmidt = true;
         istringstream is(line.substr(12));
         is >> schmidt_file;
      }else if(line.substr(0,9)=="task_prop"){
         istringstream is(line.substr(9));
         std::string property;
         while (is >> property){
            task_prop.push_back(property);
         }
      }else if(line.substr(0,13)=="restart_sweep"){
         restart_sweep = stoi(line.substr(13));
      }else if(line.substr(0,12)=="restart_bond"){
         restart_bond = stoi(line.substr(12));
      }else if(line.substr(0,9)=="timestamp"){
         timestamp = true;
      }else if(line.substr(0,8)=="alg_hvec"){
         alg_hvec = stoi(line.substr(8));
      }else if(line.substr(0,10)=="alg_hinter"){
         alg_hinter = stoi(line.substr(10));
      }else if(line.substr(0,10)=="alg_hcoper"){
         alg_hcoper = stoi(line.substr(10));
      }else if(line.substr(0,10)=="alg_renorm"){
         alg_renorm = stoi(line.substr(10));
      }else if(line.substr(0,10)=="alg_rinter"){
         alg_rinter = stoi(line.substr(10));
      }else if(line.substr(0,10)=="alg_rcoper"){
         alg_rcoper = stoi(line.substr(10));
      }else if(line.substr(0,9)=="alg_decim"){
         alg_decim = stoi(line.substr(9));
      }else if(line.substr(0,7)=="alg_rdm"){
         alg_rdm = stoi(line.substr(7));
      }else if(line.substr(0,9)=="debug_rdm"){
         debug_rdm = true;
      }else if(line.substr(0,7)=="notrunc"){
         notrunc = true;
      }else if(line.substr(0,7)=="ifdist1"){
         ifdist1 = true;	      
      }else if(line.substr(0,7)=="ifdistc"){
         ifdistc = true;	      
      }else if(line.substr(0,13)=="save_formulae"){
         save_formulae = true;
      }else if(line.substr(0,13)=="sort_formulae"){
         sort_formulae = true;
      }else if(line.substr(0,11)=="save_mmtask"){
         save_mmtask = true;
      }else if(line.substr(0,9)=="batchhvec"){
         std::string batchinter, batchgemm, batchred;
         istringstream is(line.substr(10));
         is >> batchinter >> batchgemm >> batchred;
         batchhvec = std::make_tuple(stoi(batchinter), stoi(batchgemm), stoi(batchred)); 
      }else if(line.substr(0,11)=="batchrenorm"){
         std::string batchinter, batchgemm, batchred;
         istringstream is(line.substr(11));
         is >> batchinter >> batchgemm >> batchred;
         batchrenorm = std::make_tuple(stoi(batchinter), stoi(batchgemm), stoi(batchred)); 
      }else if(line.substr(0,8)=="batchmem"){
         batchmem = stod(line.substr(8));
      }else if(line.substr(0,11)=="rcanon_file"){
         istringstream is(line.substr(11));
         is >> rcanon_file;
      }else if(line.substr(0,12)=="rcanon2_file"){
         istringstream is(line.substr(12));
         is >> rcanon2_file;
      }else if(line.substr(0,7)=="ciroots"){
         istringstream is(line.substr(7));
         string s;
         while(is>>s){
            ciroots.push_back( stoi(s) );
         }
      }else if(line.substr(0,7)=="maxdets"){
         maxdets = stoi(line.substr(7)); 
      }else if(line.substr(0,11)=="thresh_proj"){
         thresh_proj = stod(line.substr(11)); 
      }else if(line.substr(0,12)=="thresh_ortho"){
         thresh_ortho = stod(line.substr(12));
      }else if(line.substr(0,7)=="rdm_svd"){
         rdm_svd = stod(line.substr(7));
      }else if(line.substr(0,7)=="outprec"){
         outprec = stoi(line.substr(7));
      }else if(line.substr(0,6)=="nroots"){
         nroots = stoi(line.substr(6));
      }else if(line.substr(0,5)=="guess"){
         guess = stoi(line.substr(5));
      }else if(line.substr(0,6)=="iomode"){
         iomode = stoi(line.substr(6));    
      }else if(line.substr(0,7)=="ioasync"){
         async_fetch = true;
         async_save = true;
         async_remove = true;
      }else if(line.substr(0,11)=="async_fetch"){
         async_fetch = true;
      }else if(line.substr(0,10)=="async_save"){
         async_save = true;
      }else if(line.substr(0,12)=="async_remove"){
         async_remove = true;
      }else if(line.substr(0,6)=="ifnccl"){
         ifnccl = true;
      }else if(line.substr(0,5)=="iroot"){
         iroot = stoi(line.substr(5));
      }else if(line.substr(0,5)=="jroot"){
         jroot = stoi(line.substr(5));
      }else if(line.substr(0,7)=="nsample"){
         nsample = stoi(line.substr(7));
      }else if(line.substr(0,4)=="nprt"){
         nprt = stoi(line.substr(4));
      }else if(line.substr(0,5)=="pthrd"){
         pthrd = stod(line.substr(5));
      }else if(line.substr(0,11)=="thresh_cabs"){
         thresh_cabs = stod(line.substr(11));
      }else if(line.substr(0,5)=="tosu2"){
         tosu2 = true;
      }else if(line.substr(0,12)=="thresh_tosu2"){
         thresh_tosu2 = stod(line.substr(12));
      }else if(line.substr(0,7)=="singlet"){
         singlet = true;
      }else if(line.substr(0,8)=="fromconf"){
         istringstream is(line.substr(8));
         is >> inputconf;
      }else if(line.substr(0,9)=="saveconfs"){
         istringstream is(line.substr(9));
         is >> saveconfs;
      }else if(line.substr(0,9)=="loadconfs"){
         istringstream is(line.substr(9));
         is >> loadconfs;
      }else if(line.substr(0,9)=="diagcheck"){
         diagcheck = true;
      }else if(line.substr(0,7)=="savebin"){
         savebin = true;
      }else if(line.substr(0,9)=="fromnosym"){
         fromnosym = true;
      }else if(line.substr(0,8)=="cisolver"){
         cisolver = stoi(line.substr(8)); 
      }else if(line.substr(0,8)=="maxcycle"){
         maxcycle = stoi(line.substr(8));
      }else if(line.substr(0,5)=="nbuff"){
         nbuff = stoi(line.substr(5));
      }else if(line.substr(0,7)=="damping"){
         damping = stod(line.substr(7));
      }else if(line.substr(0,9)=="noprecond"){
         precond = false;
      }else if(line.substr(0,7)=="dbranch"){
         dbranch = stoi(line.substr(7));
      }else if(line.substr(0,8)=="maxsweep"){
         maxsweep = stoi(line.substr(8));
      }else if(line.substr(0,7)=="maxbond"){
         maxbond = stoi(line.substr(7));
      }else if(line.substr(0,7)=="oo_iprt"){
         ooparams.iprt = stoi(line.substr(7));
      }else if(line.substr(0,10)=="oo_maxiter"){
         ooparams.maxiter = stoi(line.substr(10));
      }else if(line.substr(0,7)=="oo_dfac"){
         ooparams.dfac = stoi(line.substr(7));
      }else if(line.substr(0,12)=="oo_macroiter"){
         ooparams.macroiter = stoi(line.substr(12));
      }else if(line.substr(0,12)=="oo_microiter"){
         ooparams.microiter = stoi(line.substr(12));
      }else if(line.substr(0,8)=="oo_alpha"){
         ooparams.alpha = stod(line.substr(8));
      }else if(line.substr(0,10)=="oo_thrdopt"){
         ooparams.thrdopt = stod(line.substr(10));
      }else if(line.substr(0,10)=="oo_thrdloc"){
         ooparams.thrdloc = stod(line.substr(10));
      }else if(line.substr(0,10)=="oo_thrddwt"){
         ooparams.thrddwt = stod(line.substr(10));
      }else if(line.substr(0,10)=="oo_thrdeps"){
         ooparams.thrdeps = stod(line.substr(10));
      }else if(line.substr(0,9)=="oo_nptloc"){
         ooparams.nptloc = stoi(line.substr(9));
      }else if(line.substr(0,12)=="oo_acceptall"){
         ooparams.acceptall = true;
      }else if(line.substr(0,15)=="oo_unrestricted"){
         ooparams.unrestricted = true;
      }else if(line.substr(0,7)=="oo_urot"){
         stringstream is(line.substr(7));
         is >> ooparams.urot; 
      }else if(line.substr(0,9)=="rcfprefix"){
         istringstream is(line.substr(9));
         is >> rcfprefix;
      }else if(line.substr(0,8)=="schedule"){
         while(true){
            line.clear();
            getline(istrm,line);
            std::cout << line << std::endl;
            if(line.empty() || line[0]=='#') continue;
            if(line.substr(0,3)=="end") break;
            istringstream is(line);
            string isweep, dots, dcut, eps, noise;
            is >> isweep >> dots >> dcut >> eps >> noise;
            tmp_ctrls.push_back({stoi(isweep),stoi(dots),stoi(dcut),stod(eps),stod(noise)});
         }
      }else{
         tools::exit("error: no matching key! line = "+line);
      }
   }
   // setup ctrls
   int size = tmp_ctrls.size();
   if(size == 0){
      if(task_opt) tools::exit("error: schedule is not specified!");
   }else{
      // consistency check
      if(tmp_ctrls[0].isweep != 0){
         tools::exit("error: schedule must start with isweep=0");
      }
      for(int i=1; i<size; i++){
         if(tmp_ctrls[i].isweep <= tmp_ctrls[i-1].isweep){
            tools::exit("error: schedule is invalid for i="+std::to_string(i)); 
         }
      }
      // put control parameters into ctrls
      ctrls.resize(maxsweep);
      for(int i=1; i<size; i++){
         for(int j=tmp_ctrls[i-1].isweep; j<tmp_ctrls[i].isweep; j++){
            if(j < maxsweep){
               ctrls[j] = tmp_ctrls[i-1];
               ctrls[j].isweep = j;
            }
         }
      }
      for(int j=tmp_ctrls[size-1].isweep; j<maxsweep; j++){
         ctrls[j] = tmp_ctrls[size-1];
         ctrls[j].isweep = j;
      }
   } // size
   // check input
   assert(alg_hcoper <= 2);
   assert(alg_rcoper <= 1);
   // diag gpu check
   if(diagcheck && async_fetch){
      std::cout << "error: diagcheck should not be used with async_fetch!" << std::endl;
      exit(1);
   }
   // nccl
   bool usecpu = (alg_hvec <= 10 || alg_renorm <=10);
   if(usecpu && ifnccl){
      std::cout << "error: ifnccl=true only for alg_hvec/alg_renorm>10 [gpu]!" << std::endl;
      exit(1);
   }
   // alg_rdm=0 and debug_rdm
   if(alg_rdm == 0 and !debug_rdm){
      std::cout << "error: alg_rdm=0 should be used with debug_rdm=true!" << std::endl;
      exit(1);
   }
}

void params_ctns::print() const{
   cout << "\n===== params_ctns::print =====" << endl;
   cout << "qkind = " << qkind << endl;
   cout << "topology_file = " << topology_file << endl;
   // debug level
   cout << "verbose = " << verbose << endl;
   // ZL@20220210 new task structure
   cout << "task_init = " << task_init << endl;
   cout << "task_sdiag = " << task_sdiag << endl;
   cout << "task_ham = " << task_ham << endl;
   cout << "task_opt = " << task_opt << endl;
   cout << "task_oodmrg = " << task_oodmrg << endl;
   cout << "task_vmc = " << task_vmc << endl;
   cout << "task_expand = " << task_expand << endl;
   cout << "task_tononsu2 = " << task_tononsu2 << endl;
   cout << "task_prop = ";
   for(const auto& k : task_prop) std::cout << k << " ";
   std::cout << std::endl;
   cout << "task_schmidt = " << task_schmidt << endl;
   cout << "schmidt_file = " << schmidt_file << endl;
   cout << "restart_sweep = " << restart_sweep << endl;
   cout << "restart_bond = " << restart_bond << endl;
   cout << "timestamp = " << timestamp << endl;
   // conversion of sci
   cout << "ciroots = ";
   for(const auto& k : ciroots) std::cout << k << " ";
   std::cout << std::endl;
   cout << "maxdets = " << maxdets << endl;
   cout << "thresh_proj = " << scientific << thresh_proj << endl;
   cout << "thresh_ortho = " << scientific << thresh_ortho << endl;
   cout << "rdm_svd = " << scientific << rdm_svd << endl;
   cout << "outprec = " << outprec << endl;
   // sweep
   cout << "nroots = " << nroots << endl;
   cout << "guess = " << guess << endl;
   cout << "dbranch = " << dbranch << endl;
   cout << "maxsweep = " << maxsweep << endl;
   cout << "maxbond = " << maxbond << endl;
   if(maxsweep > 0){
      cout << "schedule: isweep, dots, dcut, eps, noise" << endl;
      for(int i=0; i<maxsweep; i++){
         auto& tmp = ctrls[i];
         cout << tmp.isweep << " " << tmp.dots << " " 
            << tmp.dcut << " " << tmp.eps << " " 
            << tmp.noise << endl;
      } // i
   }
   cout << "rcfprefix = " << rcfprefix << endl;
   // algorithm
   cout << "alg_hvec = " << alg_hvec << endl;
   cout << "alg_hinter = " << alg_hinter << endl;
   cout << "alg_hcoper = " << alg_hcoper << endl;
   cout << "alg_renorm = " << alg_renorm << endl;
   cout << "alg_rinter = " << alg_rinter << endl;
   cout << "alg_rcoper = " << alg_rcoper << endl;
   cout << "alg_rdm = " << alg_rdm << endl;
   cout << "alg_decim = " << alg_decim << endl;
   cout << "notrunc = " << notrunc << endl;
   cout << "ifdist1 = " << ifdist1 << endl;
   cout << "ifdistc = " << ifdistc << endl;
   cout << "save_formulae = " << save_formulae << endl;
   cout << "sort_formulae = " << sort_formulae << endl;
   cout << "save_mmtask = " << save_mmtask << endl;
   cout << "batchhvec = " << std::get<0>(batchhvec)
      << " " << std::get<1>(batchhvec)
      << " " << std::get<2>(batchhvec)
      << std::endl;
   cout << "batchrenorm = " << std::get<0>(batchrenorm)
      << " " << std::get<1>(batchrenorm)
      << " " << std::get<2>(batchrenorm)
      << std::endl;
   cout << "batchmem = " << batchmem << std::endl;
   // dvdson
   cout << "cisolver = " << cisolver << endl;
   cout << "maxcycle = " << maxcycle << endl;
   cout << "nbuff = " << nbuff << endl;
   cout << "damping = " << damping << endl;
   cout << "precond = " << precond << endl;
   // io
   cout << "rcanon_file = " << rcanon_file << endl;
   cout << "rcanon2_file = " << rcanon2_file << endl;
   // oper_poll
   cout << "iomode = " << iomode << endl;
   cout << "async_fetch = " << async_fetch << endl;
   cout << "async_save = " << async_save << endl;
   cout << "async_remove = " << async_remove << endl;
   cout << "ifnccl = " << ifnccl << endl;
   // sampling
   cout << "iroot = " << iroot << endl;
   cout << "jroot = " << jroot << endl;
   cout << "nsample = " << nsample << endl;
   cout << "nprt = " << nprt << endl;
   cout << "pthrd = " << pthrd << endl;
   cout << "thresh_cabs = " << thresh_cabs << endl;
   // su2 symmetry
   cout << "tosu2 = " << tosu2 << endl;
   cout << "thresh_tosu2 = " << scientific << thresh_tosu2 << endl;
   cout << "singlet = " << singlet << endl;
   // gpu
   cout << "diagcheck = " << diagcheck << endl;
   // rdm and entropy
   cout << "savebin = " << savebin << endl;
   // nosym
   cout << "fromnosym = " << fromnosym << endl;
   // orbital optimization
   ooparams.print();
   // configurations
   cout << "inputconf = " << inputconf << endl;
   cout << "saveconfs = " << saveconfs << endl;
   cout << "loadconfs = " << loadconfs << endl;
   cout << "debug_rdm = " << debug_rdm << endl;
}
