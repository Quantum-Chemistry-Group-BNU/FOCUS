
machine = wuhan#jiageng #dell2 #dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo

ifeq ($(strip $(machine)), dell2)
   CUDA_DIR = /home/dell/anaconda3/envs/pytorch
else ifeq ($(strip $(machine)), jiageng)
   CUDA_DIR= ${CUDADIR}
else ifeq ($(strip $(machine)), wuhan)
   CUDA_DIR=/home/HPCBase/compilers/cuda/11.4.0
endif

INCS = -I ${CUDA_DIR}/include
LIBS = 

CXX = nvcc
CXXFLAGS = -O3 -std=c++11  

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) 
#LFLAGS = -shared -fPIC
#LFLAGS = -dynamiclib -fPIC # for *.dylib

OBJECTS = twodot_diagGPU_kernel.o 

all: libctnsGPU.a

libctnsGPU.a: $(OBJECTS)
	ar crv $@ $^

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@  $(INCS) $(LIBS)

clean:
	rm -f *.o *.a *.out 
