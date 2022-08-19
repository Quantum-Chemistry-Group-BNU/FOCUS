#ifndef PREPROCESS_MMREDUCE_H
#define PREPROCESS_MMREDUCE_H

namespace ctns{

    template <typename Tm>
        struct MMreduce{
            public:
                void reduction(Tm* workspace, Tm* y, const int iop);
            public:
                size_t size, ndim, offout;
                std::vector<Tm> alpha;
                std::vector<size_t> yoff;
        };

    // y[ndim] += \sum_i a_i*y_i[ndim]
    template <typename Tm>
        void MMreduce<Tm>::reduction(Tm* workspace, Tm* y, const int iop){
            std::vector<Tm*> yptr(size);
            for(int i=0; i<size; i++){
                yptr[i] = workspace + yoff[i]; 
            }
            if(iop == 0){
                for(int i=0; i<size; i++){
                    linalg::xaxpy(ndim, alpha[i], yptr[i], y+offout); 
                }
#ifdef GPU
            }else if(iop == 1){
                for(int i=0; i<size; i++){
                    linalg::xaxpy_magma(ndim, alpha[i], yptr[i], y+offout); 
                }
#endif
	    }else{
		std::cout << "error: no such option in MMreduce<Tm>::reduction iop=" << iop << std::endl;
	        exit(1);
            }
        }

} // ctns

#endif
