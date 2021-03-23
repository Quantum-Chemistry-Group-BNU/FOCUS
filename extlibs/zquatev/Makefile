INCLUDE = -I.
MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib  
LIBS = -L$(MATHLIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl 

CXX      = g++
CXXFLAGS = -O3 -DNDEBUG -std=c++11 $(INCLUDE)

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
