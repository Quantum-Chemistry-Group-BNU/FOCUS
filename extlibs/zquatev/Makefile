
machine = dell2 #scv7260 #scy0799 #scv7260 #DCU_419 #mac #lenovo
USE_ILP64 = yes

INCLUDE = -I.

LIBS =
ifeq ($(strip $(machine)), lenovo)
   CXX = icpc
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell2)
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /home/dell/intel/oneapi/mkl/2022.0.2
else ifeq ($(strip $(machine)), DCU_419)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /public/software/compiler/intel/oneapi/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), scv7260)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), scy0799)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/OneApi/2022.1/oneapi/mkl/latest/lib/intel64/
else ifeq ($(strip $(machine)), jiageng)
   CXX = g++ #icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = ${MKLROOT}
else ifeq ($(strip $(machine)), mac)
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /opt/anaconda3/envs/work/lib
   LIBS += -lm
endif

ifeq ($(strip $(USE_ILP64)), no)
   LIBS += -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread  -ldl 
else
   CXXFLAGS += -DMKL_ILP64
   LIBS += -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread  -ldl 
endif

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) $(LIBS)
#LFLAGS = -shared -fPIC
#LFLAGS = -dynamiclib -fPIC # for *.dylib

OBJECTS = zquatev.o blocked.o unblocked.o transpose.o 

all: libzquatev.a test.x

libzquatev.a: $(OBJECTS)
	ar crv $@ $^

test.x: test.o
	$(LD) $(LDFLAGS) -o $@ $^ -L. -lzquatev 

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.a test.x *.out 
