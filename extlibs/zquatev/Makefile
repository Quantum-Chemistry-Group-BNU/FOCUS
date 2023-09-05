
machine = wuhan#cluster#dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo
USE_ILP64 = yes 
USE_BLAS = yes

INCS = -I.
LIBS =
ifeq ($(strip $(machine)), lenovo)
   CXX = icpc
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell2)
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB = /home/dell/intel/oneapi/mkl/2022.0.2
else ifeq ($(strip $(machine)), DCU_419)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB = /public/software/compiler/intel/oneapi/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), scv7260)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), wuhan)
   CXX = g++ 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
	 #openblas
   MATHLIB = ~/xiangchunyang/software/OpenBLAS-0.3.23-install-ilp64/lib
	 #kblas
   #MATHLIB =/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/kblas-install-ilp64/omp
   #LAPACKLIB = ~/xiangchunyang/software/lapack-3.11.0-install-kblas64
else ifeq ($(strip $(machine)), cluster)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB =/share/app/oneapi2022/mkl/2022.0.2/lib/intel64
else ifeq ($(strip $(machine)), scy0799)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17
   MATHLIB =/data/apps/OneApi/2022.1/oneapi/mkl/latest/lib/intel64/
else ifeq ($(strip $(machine)), jiageng)
   CXX = g++ #icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB = ${MKLROOT}
else ifeq ($(strip $(machine)), mac)
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 
   MATHLIB = /opt/anaconda3/envs/work/lib
   LIBS += -lm
endif

ifeq ($(strip $(USE_BLAS)), yes)
   ifeq ($(strip $(USE_ILP64)), no)
      LIBS += -L $(MATHLIB) -lopenblas_omp  -lpthread  -ldl 
   else
      CXXFLAGS += -DLAPACK_ILP64 -DMKL_ILP64 -DOPENBLAS_USE64BITINT -DUSE64BITINT
      LIBS += -L $(MATHLIB) -lopenblas_omp  -lpthread  -ldl 
      #LIBS += -L $(MATHLIB) -lkblas -L $(LAPACKLIB) -llapack64  -lpthread  -ldl 
   endif
endif

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) 
#LFLAGS = -shared -fPIC
#LFLAGS = -dynamiclib -fPIC # for *.dylib

OBJECTS = zquatev.o blocked.o unblocked.o transpose.o 

all: libzquatev.a test.x

libzquatev.a: $(OBJECTS)
	ar crv $@ $^

test.x: test.o
	$(LD) $(LDFLAGS) -o $@ $^  $(INCS) $(LIBS) -L. -lzquatev

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@  $(INCS) $(LIBS)

clean:
	rm -f *.o *.a test.x *.out 
