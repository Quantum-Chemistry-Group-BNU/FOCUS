#ifndef CTNS_OODMRG_UROT_H
#define CTNS_OODMRG_UROT_H

namespace ctns{

   // umat contains urot_a and urot_b, which for spin-restricted rotations are the same!
   template <typename Tm> 
      struct urot_class{
         public:
            urot_class(const bool _unrestricted, const int _norb){ // no. of spatial orbitals
               unrestricted = _unrestricted;
               umat.resize(2);
               umat[0].resize(_norb,_norb); 
               umat[1].resize(_norb,_norb);
            }
            void initialize(const input::schedule& schd){
               int norb = umat[0].rows();
               if(schd.ctns.ooparams.urot.empty()){
                  umat[0] = linalg::identity_matrix<Tm>(norb);
                  umat[1] = linalg::identity_matrix<Tm>(norb);
               }else{
                  umat[0].load_txt(schd.scratch+"/"+schd.ctns.ooparams.urot+"_0");
                  umat[1].load_txt(schd.scratch+"/"+schd.ctns.ooparams.urot+"_1");
               }
               assert(linalg::check_orthogonality(umat[0]) < 1.e-8 and
                     linalg::check_orthogonality(umat[1]) < 1.e-8);
            }
            void save_txt(const std::string urot_file, const int outprec) const{
               umat[0].save_txt(urot_file+"_0", outprec); 
               umat[1].save_txt(urot_file+"_1", outprec); 
            }
         public:
            bool unrestricted = false;
            std::vector<linalg::matrix<Tm>> umat;
      };

} // ctns

#endif
