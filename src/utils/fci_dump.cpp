#include "fci.h"
#include <fstream>
#include <cmath>

using namespace std;
using namespace fock;
using namespace fci;

void sparse_hamiltonian::to_gephi(const string& fname, 
			          const onspace& space){
   cout << "\nsparse_hamiltonian::to_gephi fname = " << fname << endl;
   //
   // save 
   //
   std::ofstream file(fname+".gexf");
   file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
   file << "<gexf xmlns=\"http://www.gexf.net/1.2draft\" xmlns:viz=\"http://www.gexf.net/1.2draft/viz\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd\" version=\"1.2\">" << endl;
   file << "    <meta lastmodifieddate=\"2020-03-22\">" << endl;
   file << "        <creator>Zhendong Li</creator>" << endl;
   file << "        <description>Hamiltonian file</description>" << endl;
   file << "    </meta>" << endl;
   //
   // graph
   //
   file << "    <graph mode=\"static\" defaultedgetype=\"undirected\">" << endl;
   //
   // nodes
   //
   file << "        <nodes>" << endl;
   for(int i=0; i<dim; i++){
      file << "            <node id=\""+to_string(i)
	     +"\" label=\""+space[i].to_string2()+"\"/>" << endl;
   }
   file << "        </nodes>" << endl;
   //
   // edges
   //
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
