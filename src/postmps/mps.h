#ifndef MPS_H
#define MPS_H

namespace ctns{

   template <typename Km>
      struct mps{
         public:
            mps(){
               if(!qkind::is_available<Km>()) tools::exit("error: no such qkind for MPS!");
            }
            // load
            void load(const std::string fname){
               std::cout << "ctns::mps::load fname=" << fname << std::endl;
               std::ifstream ifs(fname, std::ios::binary);
               boost::archive::binary_iarchive load(ifs);
               // load sites
               sites.resize(nphysical);
               for(int idx=0; idx<nphysical; idx++){
                  int jdx = nphysical-1-idx; // interface to CTNS format [!!!] 
                  load >> sites[jdx];
               }
               load >> rwfuns;
               ifs.close();
            }
            // physical index
            int get_pindex(const int i) const{ return image2[2*i]/2; }
            // site0 & rwfuns
            qsym get_sym_state() const{
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns[0].info.qrow.get_sym(0);
            }
            int get_nroots() const{ 
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns.size(); 
            }
            // wf2(iroot,icol): ->-*->-
            stensor2<typename Km::dtype> get_wf2() const{
               int nroots = this->get_nroots();
               qbond qrow({{this->get_sym_state(),nroots}});
               const auto& qcol = rwfuns[0].info.qcol;
               const auto& dir = rwfuns[0].info.dir;
               assert(dir == dir_RWF);
               stensor2<typename Km::dtype> wf2(rwfuns[0].info.sym, qrow, qcol, dir);
               for(int iroot=0; iroot<nroots; iroot++){
                  for(int ic=0; ic<rwfuns[0].info.qcol.get_dim(0); ic++){
                     wf2(0,0)(iroot,ic) = rwfuns[iroot](0,0)(0,ic);
                  }
               }
               return wf2;
            }
            // site0
            stensor3<typename Km::dtype> get_site0() const{
               const auto& site0 = sites[0];
               return contract_qt3_qt2("l",site0,this->get_wf2());
            }
            // print
            void print() const{
               std::cout << "ctns::mps::print A[l,r,n]" << std::endl;
               for(int k=0; k<nphysical; k++){
                  auto shape = sites[k].get_shape();
                  std::cout << " k=" << k 
                     << " (" << std::get<0>(shape) // l 
                     << "," << std::get<1>(shape) // r
                     << "," << std::get<2>(shape) // n
                     << ")" << std::endl;
               }
            }
            // underlying hilbert space
            fock::onspace get_fcispace() const{
               fock::onspace fci_space;
               auto sym_state = this->get_sym_state();
               int isym = sym_state.isym();
               if(isym == 2){
                  int ne = sym_state.ne();
                  int tm = sym_state.tm();
                  int na = (ne+tm)/2;
                  int nb = (ne-tm)/2;
                  fci_space = fock::get_fci_space(nphysical, na, nb);
                  std::cout << "Generate Hilbert space for"
                     << " (ks,na,nb)=" << nphysical << "," << na << "," << nb
                     << " dim=" << fci_space.size()
                     << std::endl;
               }else if(isym == 1){
                  int ne = sym_state.ne();
                  fci_space = fock::get_fci_space(nphysical*2, ne);
                  std::cout << "Generate Hilbert space for" 
                     << " (k,n)=" << nphysical*2 << "," << ne
                     << " dim=" << fci_space.size()
                     << std::endl; 
               }else{
                  std::cout << "error: isym is not supported yet! isym=" << isym << endl;
                  exit(1); 
               }
               return fci_space;
            }
         public:
            using Tm = typename Km::dtype;
            int nphysical;
            std::vector<int> image2;
            std::vector<stensor3<Tm>> sites;
            std::vector<stensor2<Tm>> rwfuns;
      };

} // ctns

#ifndef SERIAL

namespace mpi_wrapper{

   // icomb: assuming the individual size of sites is small
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::mps<Tm>& imps, int root){
         boost::mpi::broadcast(comm, imps.nphysical, root);
         int rank = comm.rank();
         if(rank != root) imps.sites.resize(imps.nphysical); // reserve space
         for(int i=0; i<imps.nphysical; i++){
            boost::mpi::broadcast(comm, imps.sites[i], root);
         }
         boost::mpi::broadcast(comm, imps.rwfuns, root);
      }

} // mpi_wrapper

#endif

#endif
