
machine = dell2 #dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo

ifeq ($(strip $(machine)), dell2)
   CXX = nvcc
   CUDA_DIR = /home/dell/anaconda3/envs/pytorch
endif
CXXFLAGS = -O2

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
