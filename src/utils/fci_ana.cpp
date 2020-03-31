#include <fstream>
#include <cmath>
#include <functional>
#include "../core/matrix.h"
#include "fci.h"

using namespace std;
using namespace fock;
using namespace fci;
using namespace linalg;

// save to *.gexf for the program gephi for visualization 
void sparse_hamiltonian::save_gephi(const string& fname, 
			          const onspace& space){
   cout << "\nsparse_hamiltonian::save_gephi fname = " << fname << endl;
   
   ofstream file(fname+".gexf");
   file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
   file << "<gexf xmlns=\"http://www.gexf.net/1.2draft\" xmlns:viz=\"http://www.gexf.net/1.2draft/viz\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd\" version=\"1.2\">" << endl;
   file << "    <meta lastmodifieddate=\"2020-03-22\">" << endl;
   file << "        <creator>Zhendong Li</creator>" << endl;
   file << "        <description>Hamiltonian file</description>" << endl;
   file << "    </meta>" << endl;
   
   // graph
   file << "    <graph mode=\"static\" defaultedgetype=\"undirected\">" << endl;
   // nodes
   file << "        <nodes>" << endl;
   for(int i=0; i<dim; i++){
      file << "            <node id=\""+to_string(i)
	     +"\" label=\""+space[i].to_string2()+"\"/>" << endl;
   }
   file << "        </nodes>" << endl;
   // edges
   file << "        <edges>" << endl;
   int id = 0;
   for(int i=0; i<dim; i++){
      for(int jdx=0; jdx<connect[i].size(); jdx++){
	 int j = connect[i][jdx];
	 double val = abs(value[i][jdx]);
         file << "            <edge id=\""+to_string(id)
	        +"\" source=\""+to_string(i)
		+"\" target=\""+to_string(j)
		+"\" weight=\""+to_string(val)
		+"\"/>" << endl;
         id++;
      } // j
   } // i
   file << "        </edges>" << endl;
   file << "    </graph>" << endl;
   file << "</gexf>" << endl;
   file.close();
}

void sparse_hamiltonian::save_text(const string& fname){ 
   cout << "\nsparse_hamiltonian::save_text fname = " << fname << endl;
   matrix mat(dim,dim);
   for(int i=0; i<dim; i++){
      for(int jdx=0; jdx<connect[i].size(); jdx++){
	 int j = connect[i][jdx];
	 double val = value[i][jdx];
	 mat(i,j) = val;
	 mat(j,i) = val;
      }
   }
   mat.save_text(fname);
}

void sparse_hamiltonian::analysis(){
   cout << "\nsparse_hamiltonian::analysis" << endl;
   map<int,int,greater<int>> bucket;
   double size = 0;
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
   double nedges = dim*(dim-1)/2;
   double avc = 2*size/dim;
   cout << "dim = " << dim
   	<< "  avc = " << defaultfloat << fixed << avc
   	<< "  per = " << defaultfloat << setprecision(3) << avc/(dim-1)*100 << endl; 
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
   cout << "average size |Hij| = " << scientific << setprecision(1) << Hsum/size << endl; 
}
