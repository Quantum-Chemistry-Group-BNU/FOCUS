#include <iomanip>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <cassert>
#include "input.h"
#include "../core/tools.h"

using namespace std;
using namespace input;

const bool debug_input = false;

//
// SCI
//
void params_sci::read(ifstream& istrm){
   if(debug_input) cout << "params_sci::read" << endl; 
   run = true;
   // helpers 
   vector<int> sweep_iter; 
   vector<double> sweep_eps;
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
      }else if(line.substr(0,6)=="nroots"){
	 nroots = stoi(line.substr(6));
      }else if(line.substr(0,4)=="flip"){
         flip = true;
      }else if(line.substr(0,4)=="eps0"){
         eps0 = stod(line.substr(4)); 
      }else if(line.substr(0,7)=="maxiter"){
         maxiter = stoi(line.substr(7));
      }else if(line.substr(0,6)=="deltaE"){
         deltaE = stod(line.substr(6)); 
      }else if(line.substr(0,3)=="pt2"){
         ifpt2 = true;
      }else if(line.substr(0,4)=="eps2"){
         eps2 = stod(line.substr(4)); 
      }else if(line.substr(0,4)=="load"){
         load = true;
      }else if(line.substr(0,4)=="dets"){
	 int ndet = 0;
	 while(true){
            line.clear();	   
            getline(istrm,line);
            std::cout << line << std::endl;
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
	    det_seeds.insert(det);
	    if(ndet != det_seeds.size()) tools::exit("error: det is redundant!");
	 }
	 nseeds = det_seeds.size();
      }else if(line.substr(0,8)=="cisolver"){
         cisolver = stoi(line.substr(8)); 
      }else if(line.substr(0,8)=="maxcycle"){
         maxcycle = stoi(line.substr(8)); 
      }else if(line.substr(0,6)=="crit_v"){
         crit_v = stod(line.substr(6));
      }else if(line.substr(0,8)=="schedule"){
	 while(true){
	    line.clear();
	    getline(istrm,line);
            std::cout << line << std::endl;
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    istringstream is(line);
	    string iter, eps;
	    is >> iter >> eps;
	    sweep_iter.push_back( stoi(iter) );
	    sweep_eps.push_back( stod(eps) );
	 }
      }else if(line.substr(0,7)=="ci_file"){
         istringstream is(line.substr(7));
	 is >> ci_file;
      }else{
	 tools::exit("error: no matching key! line = "+line);
      }
   }
   // set miniter & maxiter
   int size = sweep_iter.size();
   assert(size > 0);
   miniter = sweep_iter[size-1]+1;
   assert(maxiter >= miniter);
   // put eps1 into array
   eps1.resize(maxiter);
   for(int i=1; i<size; i++){
      for(int j=sweep_iter[i-1]; j<sweep_iter[i]; j++){
	 eps1[j] = sweep_eps[i-1];
      }
   }
   for(int j=sweep_iter[size-1]; j<maxiter; j++){
      eps1[j] = sweep_eps[size-1];
   }
}

void params_sci::print() const{
   cout << "=== params_sci::print ===" << endl;
   cout << "nroots = " << nroots << endl;
   cout << "no. of unique seeds = " << nseeds << endl;
   // print det_seeds = {|det>}
   int ndet = 0;
   for(const auto& det : det_seeds){
      cout << ndet << "-th det: ";
      for(auto k : det){
         cout << k << " ";
      }
      cout << endl;
      ndet += 1;
   } // det
   cout << "flip = " << flip << endl;
   cout << "eps0 = " << eps0 << endl;
   cout << "miniter = " << miniter << endl;
   cout << "maxiter = " << maxiter << endl;
   cout << "schedule: iter eps1" << endl;
   for(int i=0; i<maxiter; i++){
      cout << i << " " << eps1[i] << endl;
   }
   cout << "convergence parameters" << endl;
   cout << "deltaE = " << deltaE << endl;
   cout << "cisolver = " << cisolver << endl;
   cout << "maxcycle = " << maxcycle << endl;
   cout << "crit_v = " << crit_v << endl;
   // pt2
   cout << "ifpt2 = " << ifpt2 << " eps2=" << eps2 << endl;
   // io
   cout << "load = " << load << endl; 
   cout << "ci_file = " << ci_file << endl; 
}

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
      }else if(line.substr(0,4)=="kind"){
         istringstream is(line.substr(4));
	 is >> kind;
      }else if(line.substr(0,4)=="task"){
         istringstream is(line.substr(4));
	 is >> task;
      }else if(line.substr(0,13)=="topology_file"){
         istringstream is(line.substr(13));
	 is >> topology_file;
      }else if(line.substr(0,11)=="rcanon_file"){
         istringstream is(line.substr(11));
	 is >> rcanon_file;
      }else if(line.substr(0,7)=="maxdets"){
         maxdets = stoi(line.substr(7)); 
      }else if(line.substr(0,11)=="thresh_proj"){
         thresh_proj = stod(line.substr(11)); 
      }else if(line.substr(0,12)=="thresh_ortho"){
         thresh_ortho = stod(line.substr(12));
      }else if(line.substr(0,6)=="nroots"){
	 nroots = stoi(line.substr(6));
      }else if(line.substr(0,5)=="guess"){
         guess = stoi(line.substr(5));
      }else if(line.substr(0,6)=="inoise"){
	 inoise = stoi(line.substr(6));
      }else if(line.substr(0,4)=="load"){
         load = true;
      }else if(line.substr(0,8)=="cisolver"){
         cisolver = stoi(line.substr(8)); 
      }else if(line.substr(0,8)=="maxcycle"){
         maxcycle = stoi(line.substr(8)); 
      }else if(line.substr(0,8)=="maxsweep"){
	 maxsweep = stoi(line.substr(8));
      }else if(line.substr(0,8)=="schedule"){
	 while(true){
	    line.clear();
	    getline(istrm,line);
            std::cout << line << std::endl;
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    istringstream is(line);
	    string iter, dots, dcut, eps, noise;
	    is >> iter >> dots >> dcut >> eps >> noise;
	    tmp_ctrls.push_back({stoi(iter),stoi(dots),stoi(dcut),stod(eps),stod(noise)});
	 }
      }else{
	 tools::exit("error: no matching key! line = "+line);
      }
   }
   istrm.close();
   // setup ctrls
   if(maxsweep > 0){
      ctrls.resize(maxsweep);
      int size = tmp_ctrls.size();
      for(int i=1; i<size; i++){
         for(int j=tmp_ctrls[i-1].isweep; j<tmp_ctrls[i].isweep; j++){
            ctrls[j] = tmp_ctrls[i-1];
            ctrls[j].isweep = j;
         }
      }
      for(int j=tmp_ctrls[size-1].isweep; j<maxsweep; j++){
         ctrls[j] = tmp_ctrls[size-1];
         ctrls[j].isweep = j;
      }
   }
}

void params_ctns::print() const{
   cout << "=== params_ctns::print ===" << endl;
   cout << "kind = " << kind << endl;
   cout << "task = " << task << endl;
   cout << "topology_file = " << topology_file << endl;
   cout << "nroots = " << nroots << endl;
   cout << "maxdets = " << maxdets << endl;
   cout << "thresh_proj = " << scientific << thresh_proj << endl;
   cout << "thresh_ortho = " << scientific << thresh_ortho << endl;
   cout << "cisolver = " << cisolver << endl;
   cout << "maxcycle = " << maxcycle << endl;
   cout << "inoise = " << inoise << endl; 
   cout << "guess = " << guess << endl;
   cout << "maxsweep = " << maxsweep << endl;
   if(maxsweep > 0){
      cout << "schedule: iter, dots, dcut, eps, noise" << endl;
      for(int i=0; i<maxsweep; i++){
         auto& tmp = ctrls[i];
         cout << tmp.isweep << " " << tmp.dots << " " 
              << tmp.dcut << " " << tmp.eps << " " 
	      << tmp.noise << endl;
      } // i
   }
   // io
   cout << "load = " << load << endl; 
   cout << "rcanon_file = " << rcanon_file << endl;
}

//
// schedule
//
void schedule::print() const{
   cout << "=== schedule::print ===" << endl;
   cout << "scratch = " << scratch << endl;
   cout << "dtype = " << dtype << endl;
   cout << "nelec = " << nelec << endl;
   cout << "twoms = " << twoms << endl;
   cout << "integral_file = " << integral_file << endl;
   if(sci.run) sci.print();
   if(ctns.run) ctns.print();
}

void schedule::read(string fname){
   cout << "\nschedule::read fname = " << fname << endl;
   ifstream istrm(fname);
   if(!istrm) {
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
   string line;
   while(!istrm.eof()){
      line.clear();	   
      getline(istrm,line);
      std::cout << line << std::endl;
      if(line.empty() || line[0]=='#'){
	 continue; // skip empty and comments    
      }else if(line.substr(0,7)=="scratch"){
         istringstream is(line.substr(7));
	 is >> scratch;
      }else if(line.substr(0,5)=="dtype"){
	 dtype = stoi(line.substr(5));
      }else if(line.substr(0,5)=="nelec"){
	 nelec = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,5)=="twoms"){
	 twoms = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,13)=="integral_file"){
         istringstream is(line.substr(13));
	 is >> integral_file;
      }else if(line.substr(0,4)=="$sci"){
	 sci.read(istrm);
      }else if(line.substr(0,5)=="$ctns"){
	 ctns.read(istrm);
      }else{
	 tools::exit("error: no matching key! line = "+line);
      }
   }
   istrm.close();
   // consistency check
   if(scratch == ".") tools::exit("error: scratch directory must be defined!");
   print();
}
