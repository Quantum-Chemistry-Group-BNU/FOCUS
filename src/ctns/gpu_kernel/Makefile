
machine = wuhan#dell2 #jiageng #dell2 #dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo

ifeq ($(strip $(machine)), dell2)
   CUDA_DIR = ${CUDADIR}
else ifeq ($(strip $(machine)), jiageng)
   CUDA_DIR= ${CUDADIR}
else ifeq ($(strip $(machine)), wuhan)
   #CUDA_DIR=/home/HPCBase/compilers/cuda/11.4.0
   CUDA_DIR=/home/HPCBase/compilers/cuda/11.7.0
   #CUDA_DIR=/home/HPCBase/compilers/cuda/11.8.0
endif

CXX = nvcc

CXXFLAGS =  -O3 -std=c++11 -I${CUDA_DIR}/include  

SRCS = twodot_diagGPU_kernel.cu onedot_diagGPU_kernel.cu

OBJS = $(SRCS:.cu=.o)

LIBRARY = libctnsGPU.a

all: $(LIBRARY)
	
%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBRARY): $(OBJS)
	ar crv $@ $^

clean:
	rm -f *.o *.a *.out 
