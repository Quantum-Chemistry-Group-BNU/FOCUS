#include <iostream>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <cassert>
#include "input.h"

using namespace std;
using namespace input;

void input::read_input(input::schedule& schd, string fname){
   cout << "\ninput::read_input fname = " << fname << endl;

   ifstream istrm(fname);
   if(!istrm) {
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
  
   vector<int> sweep_iter; // for SCI
   vector<double> sweep_eps;

   schd.scratch = ".";
   schd.nelec = 0;
   schd.twoms = 0;
   schd.nroots = 1;
   schd.nseeds = 0;
   schd.maxiter = 0;
   schd.crit_v = 1.e-6;
   schd.maxcycle = 100;
   schd.deltaE = 1.e-10;
   schd.flip = false;
   schd.ifpt2 = false;
   schd.eps2 = 1.e-8;
   schd.eps0 = 1.e-2;
   schd.integral_file = "FCIDUMP";
   schd.integral_type = 0; // =0, RHF; =1, UHF
   schd.ciload = false;
   schd.combload = false;
   schd.topology_file = "TOPOLOGY";
   schd.maxdets = 10000;
   schd.thresh_proj = 1.e-15;
   schd.thresh_ortho = 1.e-10;
   schd.maxsweep = 0;
   	 
   string line;
   while(!istrm.eof()){
      line.clear();	   
      getline(istrm,line);
      if(line.empty() || line[0]=='#'){
	 continue; // skip empty and comments    
      }else if(line.substr(0,7)=="scratch"){
         istringstream is(line.substr(7));
	 is >> schd.scratch;
      }else if(line.substr(0,5)=="nelec"){
	 schd.nelec = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,5)=="twoms"){
	 schd.twoms = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,6)=="nroots"){
	 schd.nroots = stoi(line.substr(6));
      }else if(line.substr(0,7)=="maxiter"){
         schd.maxiter = stoi(line.substr(7)); 
      }else if(line.substr(0,6)=="crit_v"){
         schd.crit_v = stod(line.substr(6));
      }else if(line.substr(0,8)=="maxcycle"){
         schd.maxcycle = stoi(line.substr(8)); 
      }else if(line.substr(0,6)=="deltaE"){
         schd.deltaE = stod(line.substr(6)); 
      }else if(line.substr(0,4)=="flip"){
         schd.flip = true;
      }else if(line.substr(0,3)=="pt2"){
         schd.ifpt2 = true;
      }else if(line.substr(0,4)=="eps2"){
         schd.eps2 = stod(line.substr(4)); 
      }else if(line.substr(0,4)=="eps0"){
         schd.eps0 = stod(line.substr(4)); 
      }else if(line.substr(0,13)=="integral_file"){
         istringstream is(line.substr(13));
	 is >> schd.integral_file;
      }else if(line.substr(0,13)=="integral_type"){
         istringstream is(line.substr(13));
	 is >> schd.integral_type;
      }else if(line.substr(0,6)=="ciload"){
         schd.ciload = true;
      }else if(line.substr(0,8)=="combload"){
         schd.combload = true;
      }else if(line.substr(0,4)=="dets"){
	 int ndet = 0;
	 while(true){
            line.clear();	   
            getline(istrm,line);
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    ndet += 1;
	    // read occ from string 
	    istringstream is(line);
	    string s;
	    set<int> det;
	    while(is>>s){
	       det.insert( stoi(s) );	    
	    }
	    schd.det_seeds.insert(det);
	    if(ndet != schd.det_seeds.size()){
	       cout << "error: det is redundant!" << endl;	    
	       exit(1);
	    }
	 }
	 schd.nseeds = schd.det_seeds.size();
      }else if(line.substr(0,12)=="sci_schedule"){
	 while(true){
	    line.clear();
	    getline(istrm,line);
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    istringstream is(line);
	    string iter, eps;
	    is >> iter >> eps;
	    sweep_iter.push_back( stoi(iter) );
	    sweep_eps.push_back( stod(eps) );
	 }
      }else if(line.substr(0,13)=="comb_schedule"){
	 while(true){
	    line.clear();
	    getline(istrm,line);
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    istringstream is(line);
	    string iter, dots, dcut, eps, noise;
	    is >> iter >> dots >> dcut >> eps >> noise;
	    schd.combsweep.push_back({stoi(iter),stoi(dots),stoi(dcut),stod(eps),stod(noise)});
	 }
      }else if(line.substr(0,13)=="topology_file"){
         istringstream is(line.substr(13));
	 is >> schd.topology_file;
      }else if(line.substr(0,7)=="maxdets"){
         schd.maxdets = stoi(line.substr(7)); 
      }else if(line.substr(0,11)=="thresh_proj"){
         schd.thresh_proj = stod(line.substr(11)); 
      }else if(line.substr(0,12)=="thresh_ortho"){
         schd.thresh_ortho = stod(line.substr(12));
      }else if(line.substr(0,8)=="maxsweep"){
	 schd.maxsweep = stoi(line.substr(8));
      }else{
         cout << "error: no matching key! line = " << line << endl;
	 exit(1);
      }
   }
   istrm.close();

   //---------------------
   // process information
   //---------------------
   cout << "scratch = " << schd.scratch << endl;
   cout << "nelec = " << schd.nelec << endl;
   assert(schd.nelec > 0);
   cout << "twoms = " << schd.twoms << endl;
   cout << "nroots = " << schd.nroots << endl;
   assert(schd.nroots > 0);
   cout << "integral_file = " << schd.integral_file << endl;
   cout << "integral_type = " << schd.integral_type << endl;
   cout << "no. of unique seeds = " << schd.nseeds << endl;
   int ndet = 0;
   for(const auto& det : schd.det_seeds){
      cout << ndet << "-th det: ";
      int nelec = 0; 
      for(auto k : det){
         cout << k << " ";
	 nelec += 1;
      }
      cout << endl;
      if(nelec != schd.nelec){
         cout << "error: det is not consistent with nelec!" << endl;
	 exit(1);
      }
      ndet += 1;
   } // det
   cout << "flip = " << schd.flip << endl;
   cout << "eps0 = " << schd.eps0 << endl;
   // set miniter & maxiter
   int size = sweep_iter.size();
   assert(size > 0);
   schd.miniter = sweep_iter[size-1]+1;
   cout << "miniter = " << schd.miniter << endl;
   cout << "maxiter = " << schd.maxiter << endl;
   assert(schd.maxiter >= schd.miniter);
   // put eps1 into array
   schd.eps1.resize(schd.maxiter);
   for(int i=1; i<size; i++){
      for(int j=sweep_iter[i-1]; j<sweep_iter[i]; j++){
	 schd.eps1[j] = sweep_eps[i-1];
      }
   }
   for(int j=sweep_iter[size-1]; j<schd.maxiter; j++){
      schd.eps1[j] = sweep_eps[size-1];
   }
   cout << "schedule: iter eps1" << endl;
   for(int i=0; i<schd.maxiter; i++){
      cout << i << " " << schd.eps1[i] << endl;
   }
   cout << "convergence parameters" << endl;
   cout << "deltaE = " << schd.deltaE << endl;
   cout << "dvdson:crit_v = " << schd.crit_v << endl;
   cout << "dvdson:maxcycle = " << schd.maxcycle << endl;
   // pt2
   cout << "pt2 = " << schd.ifpt2 << " eps2=" << schd.eps2 << endl;
   // comb tensor network
   cout << "topology_file = " << schd.topology_file << endl;
   cout << "maxdets = " << schd.maxdets << endl;
   cout << "thresh_proj = " << scientific << schd.thresh_proj << endl;
   cout << "thresh_ortho = " << scientific << schd.thresh_ortho << endl;
   // sweep
   cout << "maxsweep = " << schd.maxsweep << endl;
   if(schd.maxsweep > 0) init_combsweep(schd.maxsweep, schd.combsweep);
}

void input::init_combsweep(const int maxsweep,
			   vector<sweep_ctrl>& combsweep){
   auto tmp = combsweep;
   combsweep.resize(maxsweep);
   int size = tmp.size();
   for(int i=1; i<size; i++){
      for(int j=tmp[i-1].isweep; j<tmp[i].isweep; j++){
	 combsweep[j] = tmp[i-1];
	 combsweep[j].isweep = j;
      }
   }
   for(int j=tmp[size-1].isweep; j<maxsweep; j++){
      combsweep[j] = tmp[size-1];
      combsweep[j].isweep = j;
   }
   // check
   cout << "comb_schedule: iter, dots, dcut, eps, noise" << endl;
   for(int i=0; i<maxsweep; i++){
      auto ctrl = combsweep[i];
      cout << ctrl.isweep << " " 
	   << ctrl.dots << " " 
	   << ctrl.dcut << " "
	   << ctrl.eps << " " 
	   << ctrl.noise
	   << endl;
   }
}
