
machine = mac #lenovo

INCLUDE = -I.
ifeq ($(strip $(machine)), lenovo)
   CXX = icpc
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else ifeq ($(strip $(machine)), dell)
   CXX = icpc 
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
else
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++17 $(INCLUDE)
   MATHLIB = /opt/anaconda3/envs/work/lib
endif

LIBS = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl 
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
	rm -f *.o *.a test.x 
