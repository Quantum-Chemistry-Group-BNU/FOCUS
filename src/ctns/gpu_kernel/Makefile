
machine = jiageng #dell2 #dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo

ifeq ($(strip $(machine)), dell2)
   CUDA_DIR = /home/dell/anaconda3/envs/pytorch
else ifeq ($(strip $(machine)), jiageng)
   CUDA_DIR= ${CUDADIR}
endif
CXX = nvcc
CXXFLAGS = -O3 -std=c++11 -I${CUDA_DIR}/include 

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) $(LIBS)
#LFLAGS = -shared -fPIC
#LFLAGS = -dynamiclib -fPIC # for *.dylib

OBJECTS = twodot_diagGPU_kernel.o 

all: libctnsGPU.a

libctnsGPU.a: $(OBJECTS)
	ar crv $@ $^

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.a *.out 
