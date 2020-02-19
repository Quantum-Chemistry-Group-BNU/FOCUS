USE_MPI = no
USE_INTEL = no #yes
EIGEN=/usr/local/Cellar/eigen/3.3.7/include/eigen3
BOOST=/usr/local

FLAGS  = -std=c++11 -g -O0 -Wall -I${EIGEN} -I${BOOST}/include ${INCLUDE_DIR} 
DFLAGS = -std=c++11 -g -O0 -Wall -I${EIGEN} -I${BOOST}/include ${INCLUDE_DIR} -DComplex
                 
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
SRC_DIR2 = ./src/utils
INCLUDE_DIR = -I$(SRC_DIR0) \
	      -I$(SRC_DIR1) \
	      -I$(SRC_DIR2)
SRC_DEP = $(wildcard $(SRC_DIR0)/*.cpp \
  	      	     $(SRC_DIR1)/*.cpp \
	  	     $(SRC_DIR2)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))
# all the files
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard src/sci/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend $(BIN_DIR)/fci.x $(BIN_DIR)/sci.x 

depend:
	set -e; \
	mkdir -p obj; \
	echo $(SRC_ALL); \
	$(CXX) -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

$(BIN_DIR)/fci.x: $(OBJ_DIR)/fci.o $(OBJ_DEP)
	@mkdir -p bin
	@echo "\n=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(OBJ_DEP)
	@mkdir -p bin
	@echo "\n=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/fci.x
	rm -f bin/sci.x
