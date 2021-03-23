
## eigen
#MATH = -I /usr/local/include/eigen3 -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE \
       -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
## serial version of MKL
#MATH = -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
# parallel version of MKL
MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib
MATH = -L$(MATHLIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl \
       -L./extlibs/zquatev -lzquatev 
#MATH = -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl \
#       -L./extlibs/zquatev -lzquatev 
# mac framework Accelerate
#MATH = -llapack -lblas

BOOST = /usr/local
FLAGS = -DGNU -std=c++11 -g -O0 -Wall ${MATH} -I${BOOST}/include ${INCLUDE_DIR} 
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
SRC_DIR_CORE = ./$(SRC)/core
SRC_DIR_CI   = ./$(SRC)/ci
SRC_DIR_TNS  = ./$(SRC)/tns
SRC_DIR_CTNS = ./$(SRC)/ctns
SRC_DIR_IO   = ./$(SRC)/io
SRC_DIR_TEST = ./$(SRC)/tests
INCLUDE_DIR = -I$(SRC_DIR_CORE) \
	      -I$(SRC_DIR_CI) \
	      -I$(SRC_DIR_TNS) \
	      -I$(SRC_DIR_CTNS) \
	      -I$(SRC_DIR_IO) \
	      -I$(SRC_DIR_TEST)
SRC_DEP = $(wildcard $(SRC_DIR_CORE)/*.cpp \
	  	     $(SRC_DIR_CI)/*.cpp \
	  	     $(SRC_DIR_TNS)/*.cpp \
	  	     $(SRC_DIR_CTNS)/*.cpp \
	  	     $(SRC_DIR_IO)/*.cpp \
	  	     $(SRC_DIR_TEST)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# all the files with main functions
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./$(SRC)/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend \
     $(BIN_DIR)/tests.x \
     $(BIN_DIR)/tns.x \
     $(BIN_DIR)/ctns.x

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

$(BIN_DIR)/tns.x: $(OBJ_DIR)/main_tns.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/ctns.x: $(OBJ_DIR)/main_ctns.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f *.depend
