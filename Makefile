BOOST = /usr/local

## eigen
#MATH = -I /usr/local/include/eigen3 -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE \
       -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
## serial version
#MATH = -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
# parallel version
#MATH = -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl 
## my compiled lapack/blas from netlib
MATH = -llapack -lblas

#FLAGS = -DGNU -std=c++11 -g -O0 -Wall ${MATH} -I${BOOST}/include ${INCLUDE_DIR} 
FLAGS = -DGNU -std=c++11 -g -O3 -Wall ${MATH} -I${BOOST}/include ${INCLUDE_DIR} 
#FLAGS = -DGNU -DNDEBUG -std=c++11 -g -O3 -Wall ${MATH} -I${BOOST}/include ${INCLUDE_DIR} 
#	-Wl,-no_pie -lprofiler # google profiler

USE_MPI = no
ifeq ($(USE_MPI), yes) 
	CXX = mpig++
	CC = mpigcc
	LFLAGS += -L${BOOST}/lib -lboost_serialization -lboost_mpi #-lrt
else
	CXX = g++
	CC = gcc
	LFLAGS = -L${BOOST}/lib -lboost_serialization -lboost_system -lboost_filesystem 	
endif

SRC = src
BIN_DIR = ./bin
OBJ_DIR = ./obj
# all dependence
SRC_DIR1 = ./$(SRC)/io
SRC_DIR2 = ./$(SRC)/core
SRC_DIR3 = ./$(SRC)/tests
SRC_DIR4 = ./$(SRC)/utils
INCLUDE_DIR = -I$(SRC_DIR1) \
	      -I$(SRC_DIR2) \
	      -I$(SRC_DIR3) \
	      -I$(SRC_DIR4)
SRC_DEP = $(wildcard $(SRC_DIR1)/*.cpp \
	  	     $(SRC_DIR2)/*.cpp \
	  	     $(SRC_DIR3)/*.cpp \
	  	     $(SRC_DIR4)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# all the files
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./$(SRC)/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend \
     $(BIN_DIR)/tests.x \

depend:
	set -e; \
	mkdir -p $(BIN_DIR) $(OBJ_DIR); \
	echo $(SRC_ALL); \
	$(CXX) -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

$(BIN_DIR)/tests.x: $(OBJ_DIR)/main_tests.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
