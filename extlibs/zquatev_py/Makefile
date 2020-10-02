INCLUDE  = -I.
LIBS = -L../../libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl 

CXX      = g++
CXXFLAGS = -O3 -DNDEBUG -std=c++11 $(INCLUDE)
PYBINDFLAGS = -undefined dynamic_lookup `python -m pybind11 --includes`

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) $(LIBS)
LFLAGS   = -shared -fPIC

OBJECTS = zquatev.o blocked.o unblocked.o transpose.o wrapper.o

all: libzquatev.a libzquatev.so test.x

libzquatev.a: $(OBJECTS)
	ar crv $@ $^

#test.x: test.o
#	$(LD) $(LDFLAGS) -o $@ $^ -L. -lzquatev 

test.x: libzquatev.so test.o
	$(LD) $(LDFLAGS) $^ -o $@ -L/Users/zhendongli/anaconda2/envs/py35/lib $(shell python-config --libs) 

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBINDFLAGS) -o $@ -c $< 

libzquatev.so: $(OBJECTS)
	$(LD) $(LFLAGS) $(LDFLAGS) $(PYBINDFLAGS) -o $@ $(OBJECTS)

clean:
	-rm -f *.o test.x *.so

.SUFFIXES:
.SUFFIXES: .cc .o

