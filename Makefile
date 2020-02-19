USE_MPI = no
USE_INTEL = no #yes
EIGEN=/usr/local/Cellar/eigen/3.3.7/include/eigen3
BOOST=/usr/local

FLAGS = -std=c++11 -g  -O3 -I${EIGEN} -I${BOOST}/include ${INCLUDE_sci} 
DFLAGS = -std=c++11 -g -O3 -I${EIGEN} -I${BOOST}/include ${INCLUDE_sci} -DComplex
                 
FLAGS += -fopenmp
DFLAGS +=  -fopenmp
ifeq ($(USE_MPI), yes) 
	CXX = mpig++
	CC = mpigcc
	LFLAGS += -L${BOOST}/lib -lboost_serialization -lboost_mpi #-lrt
else
	CXX = g++
	CC = gcc
	LFLAGS = -L${BOOST}/lib -lboost_serialization #-lrt
	FLAGS += -DSERIAL 
	DFLAGS += -DSERIAL 
endif

BIN_DIR = ./bin
OBJ_DIR = ./obj
SRC_DIR0 = ./src/settings
SRC_DIR1 = ./src/io
SRC_DIR2 = ./src/sci

SRC_sci = $(wildcard $(SRC_DIR0)/*.cpp \
	  	     $(SRC_DIR1)/*.cpp \
		     $(SRC_DIR2)/*.cpp)
OBJ_sci = $(patsubst %.cpp,${OBJ_DIR}/%.o,$(notdir ${SRC_sci})) 
INCLUDE_sci = -I$(SRC_DIR0) -I$(SRC_DIR1)

all: depend bin/sci.x

depend:
	set -e; \
	mkdir -p obj; \
	$(CXX) -MM $(SRC_sci) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

bin/sci.x: $(OBJ_sci)
	@mkdir -p bin
	@echo "\n=== LINK $@"
	@echo "OBJ_sci=" $(OBJ_sci)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(OBJ_DIR)/%.o:
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f bin/sci.x
	rm -f obj/*.o
