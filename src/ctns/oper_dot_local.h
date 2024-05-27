#ifndef OPER_DOT_LOCAL_H
#define OPER_DOT_LOCAL_H

#include "init_phys.h"

namespace ctns{

   template <typename Tm>
      stensor2<Tm> get_dot_opC(const int isym, const int ispin){
         auto qphys = get_qbond_phys(isym);
         auto sym = get_qsym_opC(isym, ispin);
         stensor2<Tm> op(sym,qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         if(ispin == 0){
            // c[0] = kA^+
            // [[0. 0. 0. 0.]
            //  [0. 0. 0. 1.]
            //  [1. 0. 0. 0.]
            //  [0. 0. 0. 0.]]
            mat(1,3) = 1;
            mat(2,0) = 1;
         }else{
            // also store c[1] = kB^+
            // [[ 0.  0.  0.  0.]
            //  [ 0.  0. -1.  0.]
            //  [ 0.  0.  0.  0.]
            //  [ 1.  0.  0.  0.]]
            mat(1,2) = -1;
            mat(3,0) = 1;
         }
         op.from_matrix(mat);
         return op;
      }

   template <typename Tm>
      stensor2<Tm> get_dot_opA(const int isym){
         auto qphys = get_qbond_phys(isym);
         auto sym = get_qsym_opA(isym, 1, 0);
         stensor2<Tm> op(sym,qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         // c[0].dot(c[1])
         // [[0. 0. 0. 0.]
         //  [1. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         mat(1,0) = 1;
         op.from_matrix(mat);
         return op;
      }

   // B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
   template <typename Tm>
      stensor2<Tm> get_dot_opB(const int isym, const int ispin, const int jspin){
         auto qphys = get_qbond_phys(isym);
         qsym sym = get_qsym_opB(isym, ispin, jspin);
         stensor2<Tm> op(sym,qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         if(ispin == 0 and jspin == 0){
            // aa
            // c[0].dot(a[0])
            // [[0. 0. 0. 0.]
            //  [0. 1. 0. 0.]
            //  [0. 0. 1. 0.]
            //  [0. 0. 0. 0.]]
            mat(1,1) = 1;
            mat(2,2) = 1;
         }else if(ispin == 0 and jspin == 1){
            // ab
            // c[0].dot(a[1])
            // [[0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 1.]
            //  [0. 0. 0. 0.]]
            mat(2,3) = 1;
         }else if(ispin == 1 and jspin == 0){
            // b+a = (a+b)^+
            mat(3,2) = 1;
         }else if(ispin == 1 and jspin == 1){
            // Bbb:
            // c[1].dot(a[1])
            // [[0. 0. 0. 0.]
            //  [0. 1. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 1.]]
            mat(1,1) = 1;
            mat(3,3) = 1;
         }
         op.from_matrix(mat);
         return op;
      }

   template <typename Tm>
      stensor2<Tm> get_dot_opDabba(const int isym){
         auto qphys = get_qbond_phys(isym);
         stensor2<Tm> op(qsym(isym,0,0),qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         mat(1,1) = 1;
         op.from_matrix(mat);
         return op;
      }

   template <typename Tm>
      stensor2<Tm> get_dot_opTaba(const int isym){
         auto qphys = get_qbond_phys(isym);
         auto sym = get_qsym_opS(isym,1);
         stensor2<Tm> op(qsym(isym,0,0),qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         // c[0].dot(a[1].dot(a[0]))
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]]
         mat(2,1) = 1;
         op.from_matrix(mat);
         return op;
      }

   template <typename Tm>
      stensor2<Tm> get_dot_opTbba(const int isym){
         auto qphys = get_qbond_phys(isym);
         auto sym = get_qsym_opS(isym,0);
         stensor2<Tm> op(qsym(isym,0,0),qphys,qphys);
         linalg::matrix<Tm> mat(4,4);
         // c[1].dot(a[1].dot(a[0]))
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 1. 0. 0.]]
         mat(3,1) = 1;
         op.from_matrix(mat);
         return op;
      }

} // ctns

#endif
