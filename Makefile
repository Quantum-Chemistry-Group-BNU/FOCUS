BOOST = /usr/local
MATH = -I /usr/local/include/eigen3 -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE \
       -L./libmkl -Wl,-rpath,./libmkl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
FLAGS  = -std=c++11 -g -O0 -Wall ${MATH} -I${BOOST}/include ${INCLUDE_DIR}  

USE_MPI = no
ifeq ($(USE_MPI), yes) 
	CXX = mpig++
	CC = mpigcc
	LFLAGS += -L${BOOST}/lib -lboost_serialization -lboost_mpi #-lrt
else
	CXX = g++
	CC = gcc
	LFLAGS = -L${BOOST}/lib -lboost_serialization #-lrt
endif

BIN_DIR = ./bin
OBJ_DIR = ./obj
SRC_DIR0 = ./src/settings
SRC_DIR1 = ./src/io
SRC_DIR2 = ./src/utils
SRC_DIR3 = ./src/tests
INCLUDE_DIR = -I$(SRC_DIR0) \
	      -I$(SRC_DIR1) \
	      -I$(SRC_DIR2) \
	      -I$(SRC_DIR3)
SRC_DEP = $(wildcard $(SRC_DIR0)/*.cpp \
  	      	     $(SRC_DIR1)/*.cpp \
	  	     $(SRC_DIR2)/*.cpp \
	  	     $(SRC_DIR3)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# all the files
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./src/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend \
     $(BIN_DIR)/fci.x \
     $(BIN_DIR)/sci.x \
     $(BIN_DIR)/tests.x \

depend:
	set -e; \
	mkdir -p $(BIN_DIR) $(OBJ_DIR); \
	echo $(SRC_ALL); \
	$(CXX) -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

$(BIN_DIR)/tests.x: $(OBJ_DIR)/tests.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/fci.x: $(OBJ_DIR)/fci.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
