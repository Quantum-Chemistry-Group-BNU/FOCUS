
machine = mac #lenovo

INCLUDE = -I.
ifeq ($(machine), lenovo)
   CXX = icpc
   CXXFLAGS = -O2 -DNDEBUG -std=c++11 $(INCLUDE)
   MATHLIB = /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64 
else
   CXX = g++
   CXXFLAGS = -O2 -DNDEBUG -std=c++11 $(INCLUDE)
   #MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib  
   MATHLIB = /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64 
endif

LIBS = -L$(MATHLIB) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl 
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
