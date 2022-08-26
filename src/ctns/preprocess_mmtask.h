#ifndef PREPROCESS_MMTASK_H
#define PREPROCESS_MMTASK_H

#include "preprocess_hxlist.h"
#include "preprocess_mmbatch.h"
#include "preprocess_mmreduce.h"

namespace ctns{

    // Interface to Hxlist
    template <typename Tm>
        struct MMtask{
            public:
                void init(const Hxlist<Tm>& Hxlst, 
                        const int _batchgemm,
                        const size_t _batchsize,
                        const size_t offset,
                        const int hdxorder,
                        const int icase=0);
                void kernel(const int k, Tm** ptrs){
                    // perform GEMMs [c2,c1,r,l]
                    for(int i=0; i<mmbatch2[k].size(); i++){
                    //std::cout<<"mmbatch2 k="<<k<<"; i="<<i<<std::endl;
                        mmbatch2[k][i].kernel(batchgemm, ptrs);
                    //std::cout<<"mmbatch2 1"<<std::endl;
                    }
                }
                void reduction(const int k, Tm* workspace, Tm* y, const int icase){
                    mmreduce[k].reduction(workspace, y, icase);
                }
            public:
                int batchgemm;
                size_t totsize, batchsize, nbatch;
                std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
                std::vector<MMreduce<Tm>> mmreduce; // mmreduce[ibatch]
        };
    template <typename Tm>
        using MMtasks = std::vector<MMtask<Tm>>;

    template <typename Tm>
        void MMtask<Tm>::init(const Hxlist<Tm>& Hxlst,
                const int _batchgemm,
                const size_t _batchsize,
                const size_t offset,
                const int hxorder,
                const int icase){
            batchgemm = _batchgemm;
            batchsize = _batchsize;
            totsize = Hxlst.size();
            nbatch = totsize/batchsize;
            if(totsize%batchsize != 0) nbatch += 1;
            mmbatch2.resize(nbatch);
            mmreduce.resize(nbatch);
            // process Hxlst
            for(int k=0; k<nbatch; k++){
                size_t off = k*batchsize;
                int jlen = std::min(totsize-off, batchsize);
                
                // initialization
                int nd = (icase==0)? 4 : 8;
                int pos[4] = {0,1,2,3};
                std::vector<size_t> dims(nd,0);
                // count how many gemms in each case 
                for(int j=0; j<jlen; j++){
                    int jdx = off+j;
                    auto& Hxblk = Hxlst[jdx];
                    if(icase == 1){
                        pos[0] = Hxblk.dagger[3]? 0 : 1;
                        pos[1] = Hxblk.dagger[2]? 2 : 3;
                        pos[2] = Hxblk.dagger[1]? 4 : 5;
                        pos[3] = Hxblk.dagger[0]? 6 : 7;
                    }
                    dims[pos[0]] += Hxblk.identity(3)? 0 : 1; 
                    dims[pos[1]] += Hxblk.identity(2)? 0 : Hxblk.dimout[3]; // c1
                    dims[pos[2]] += Hxblk.identity(1)? 0 : Hxblk.dimout[3]*Hxblk.dimout[2]; // r
                    dims[pos[3]] += Hxblk.identity(0)? 0 : 1; // l
                }
         
                // setup mmbatch2[k]
                // generation of mmlst2
                MMlist2<Tm> mmlst2(nd);
                for(int i=0; i<nd; i++){
                    mmlst2[i].resize(dims[i]); // c2,c1,r,l
                }
                std::vector<size_t> idx(nd,0);
                for(int j=0; j<jlen; j++){
                    int jdx = off+j;
                    auto& Hxblk = Hxlst[jdx];
                    if(icase == 1){
                        pos[0] = Hxblk.dagger[3]? 0 : 1;
                        pos[1] = Hxblk.dagger[2]? 2 : 3;
                        pos[2] = Hxblk.dagger[1]? 4 : 5;
                        pos[3] = Hxblk.dagger[0]? 6 : 7;
                    }
                    MMlist2<Tm> mmtmp2(4);
                    Hxblk.get_MMlist_twodot(mmtmp2, j*offset);
                    for(int i=0; i<mmtmp2.size(); i++){
                        int ipos = pos[i];
                        // copy the mmlst to the correct place
                        for(int k=0; k<mmtmp2[i].size(); k++){
                            mmlst2[ipos][idx[ipos]] = mmtmp2[i][k];
                            idx[ipos]++;
                        } //k
                    } // i
                } // j
                // convert to batch list
                mmbatch2[k].resize(nd);
                for(int i=0; i<nd; i++){
                    if(hxorder == 1){ // sort by cost
                        std::stable_sort(mmlst2[i].begin(), mmlst2[i].end(),
                                [](const MMinfo<Tm>& mm1, const MMinfo<Tm>& mm2){
                                return mm1.cost() > mm2.cost();
                                });
                    }

                    mmbatch2[k][i].init(mmlst2[i]);
                } // i
                
                // setup mmreduce[k]
                mmreduce[k].size = jlen;
                mmreduce[k].ndim = Hxlst[off].size;
                mmreduce[k].offout = Hxlst[off].offout;
                mmreduce[k].alpha.resize(jlen);
                mmreduce[k].yoff.resize(jlen);
                for(int j=0; j<jlen; j++){
                    int jdx = off+j;
                    auto& Hxblk = Hxlst[jdx];
                    mmreduce[k].alpha[j] = Hxblk.coeff;
                    mmreduce[k].yoff[j] = j*offset+Hxblk.offres;
                }
 
            } // k
        }

} // ctns

#endif
