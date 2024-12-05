
machine = jinan #mac # cluster#dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo

USE_GCC = yes
USE_OPENMP = yes
USE_MKL = yes
USE_ILP64 = yes # by default

INCLUDE = -I.

LIBS =
ifeq ($(strip $(machine)), lenovo)
   #CXX = icpc
   #FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   #MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
   CXX = g++ #icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = ${MKLROOT}

else ifeq ($(strip $(machine)), jinan)
   CXX = g++
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = ${MKLROOT}/lib

else ifeq ($(strip $(machine)), dell)
   CXX = icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell2)
   CXX = g++
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /home/dell/intel/oneapi/mkl/2022.0.2
else ifeq ($(strip $(machine)), DCU_419)
   CXX = icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /public/software/compiler/intel/oneapi/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), scv7260)
   CXX = icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), wuhan)
   CXX = g++ 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/OpenBLAS-0.3.23-install-ilp64/lib
   #MATHLIB =/home/HPCBase/libs/openblas0.3.18_kgcc9.3.1/lib
else ifeq ($(strip $(machine)), cluster)
   CXX = icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/share/app/oneapi2022/mkl/2022.0.2/lib/intel64
else ifeq ($(strip $(machine)), scy0799)
   CXX = icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/OneApi/2022.1/oneapi/mkl/latest/lib/intel64/
else ifeq ($(strip $(machine)), jiageng)
   CXX = g++ #icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = ${MKLROOT}

else ifeq ($(strip $(machine)), a800)
   CXX = g++ #icpc 
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = ${MKLROOT}

else ifeq ($(strip $(machine)), mac)
   CXX = g++
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   #MATHLIB = /Users/zhendongli/Desktop/FOCUS/mathlib/lapack-3.12.0/build/lib
   MATHLIB = /Users/zhendongli/anaconda3/envs/osx64test/lib
   LIBS += -lm
else ifeq ($(strip $(machine)), archlinux)
   CXX = g++
   FLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /opt/intel/oneapi/mkl/2023.1.0/lib/intel64
   LIBS += -lm
endif

# special treatment for my mac machine
ifeq ($(strip $(machine)), mac)
   #MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
   #       -lblas64 -llapack64 -lpthread -lgfortran
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl
   FLAGS += -DUSE_MKL -DMKL_ILP64 -m64 -fopenmp 
else
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.sl42kc
ifeq ($(strip $(USE_MKL)),yes)
	# FLAGS
   FLAGS += -DUSE_MKL
   ifeq ($(strip $(USE_ILP64)),yes)
      FLAGS += -DMKL_ILP64 -m64
   endif
	# LFLAGS
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB)
   ifeq ($(strip $(USE_ILP64)),yes)
      MATH += -lmkl_intel_ilp64 
   else
      MATH += -lmkl_intel_lp64 
   endif
	# OpenMP & MKL
   ifeq ($(strip $(USE_OPENMP)),no)
      MATH += -lmkl_sequential 
   else
      ifeq ($(strip $(USE_GCC)),yes)
         FLAGS += -fopenmp 
         MATH += -lmkl_gnu_thread -lgomp
      else
         FLAGS += -qopenmp 
         MATH += -lmkl_intel_thread -liomp5 
      endif
   endif
   MATH += -lmkl_core -lpthread -lm -ldl
else
   # openblas/kblas
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lopenblas -lpthread -lm -ldl -lrt 
          #-lblas64 -llapack64 -lpthread -lm -ldl -lrt 
   FLAGS += -fopenmp -DLAPACK_ILP64 -DMKL_ILP64 -DOPENBLAS_USE64BITINT -DUSE64BITINT
endif
endif
LIBS += ${MATH}

LD       = $(CXX)
LDFLAGS  = $(FLAGS) $(LIBS)
#LFLAGS = -shared -fPIC
#LFLAGS = -dynamiclib -fPIC # for *.dylib

OBJECTS = zquatev.o blocked.o unblocked.o transpose.o 

all: libzquatev.a test.x

libzquatev.a: $(OBJECTS)
	ar crv $@ $^

test.x: test.o
	$(LD) -o $@ $^ $(LDFLAGS) -L. -lzquatev 

%.o: %.cc
	$(CXX) $(FLAGS) -c $< -o $@

clean:
	rm -f *.o *.a test.x *.out 
