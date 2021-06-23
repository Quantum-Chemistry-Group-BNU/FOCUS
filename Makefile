
machine = mac #lenovo

DEBUG = yes
USE_GCC = yes
USE_OPENMP = no
USE_MPI = yes

ifeq ($(machine), lenovo)
   MATHLIB = /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64
   #BOOST = /home/lx/software/boost/install_1_59_0
   BOOST = /home/lx/software/boost/install_1_75_0
   LFLAGS = -L${BOOST}/lib -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_filesystem-mt-x64 
   ifeq ($(USE_MPI) ,yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else
   MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib
   BOOST = /usr/local
   LFLAGS = -L${BOOST}/lib -lboost_serialization -lboost_system -lboost_filesystem 
   ifeq ($(USE_MPI) ,yes)   
      LFLAGS += -lboost_mpi
   endif
endif

# quaternion matrix diagonalization
MATH = -L./extlibs/zquatev -lzquatev
ifeq ($(USE_OPENMP), no)
   # serial version of MKL
   MATH += -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
           -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
   # mac framework Accelerate
   #MATH = -llapack -lblas 
else
   ifeq ($(USE_GCC), yes)
      FLAGS += -fopenmp
   else
      FLAGS += -qopenmp	
   endif
   # https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html	
   # parallel version of MKL
   MATH += -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
   	   -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl \
   	   -liomp5
   # Use GNU OpenMP library: -lmkl_gnu_thread -lgomp replace -liomp5
endif
LFLAGS += ${MATH}

ifeq ($(USE_GCC), yes)
   # GCC compiler
   ifeq ($(DEBUG), yes)
      FLAGS += -DGNU -std=c++11 -g -O0 -Wall -I${BOOST}/include ${INCLUDE_DIR}
   else
      FLAGS += -DGNU -DNDEBUG -std=c++11 -g -O2 -Wall -I${BOOST}/include ${INCLUDE_DIR}
   endif
   ifeq ($(USE_MPI), no)
      CXX = g++
      CC = gcc
      FLAGS += -DSERIAL
   else
      CXX = mpicxx
      CC = mpicc
   endif
else
   # Intel compiler
   ifeq ($(DEBUG), yes)
      FLAGS += -std=c++11 -g -O0 -Wall -I${BOOST}/include ${INCLUDE_DIR}
   else 
      FLAGS += -DNDEBUG -std=c++11 -g -O2 -Wall -I${BOOST}/include ${INCLUDE_DIR}
   endif 
   ifeq ($(USE_MPI), no)
      CXX = icpc
      CC = icc
      FLAGS += -DSERIAL
   else
      CXX = mpiicpc
      CC = mpiicc
   endif
endif

SRC = src
BIN_DIR = ./bin
OBJ_DIR = ./obj

# all dependence
SRC_DIR_CORE = ./$(SRC)/core
SRC_DIR_IO   = ./$(SRC)/io
SRC_DIR_CI   = ./$(SRC)/ci
SRC_DIR_QT   = ./$(SRC)/ctns/qtensor
SRC_DIR_CTNS = ./$(SRC)/ctns
INCLUDE_DIR = -I$(SRC_DIR_CORE) \
	      -I$(SRC_DIR_IO) \
	      -I$(SRC_DIR_CI) \
	      -I$(SRC_DIR_QT) \
	      -I$(SRC_DIR_CTNS)
SRC_DEP = $(wildcard $(SRC_DIR_CORE)/*.cpp \
	  	     $(SRC_DIR_IO)/*.cpp  \
	  	     $(SRC_DIR_CI)/*.cpp \
	  	     $(SRC_DIR_QT)/*.cpp \
	  	     $(SRC_DIR_CTNS)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# all the files with main functions
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./$(SRC)/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend \
     $(BIN_DIR)/tests_core.x \
     $(BIN_DIR)/tests_ci.x \
     $(BIN_DIR)/tests_ctns.x \
     $(BIN_DIR)/sci.x \
     $(BIN_DIR)/ctns.x 

depend:
	set -e; \
	mkdir -p $(BIN_DIR) $(OBJ_DIR); \
	echo $(SRC_ALL); \
	$(CXX) -I${BOOST}/include -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	echo 'finish dependency check!'; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

# Executables
$(BIN_DIR)/tests_core.x: $(OBJ_DIR)/tests_core.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/tests_ci.x: $(OBJ_DIR)/tests_ci.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/tests_ctns.x: $(OBJ_DIR)/tests_ctns.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

$(BIN_DIR)/ctns.x: $(OBJ_DIR)/ctns.o $(OBJ_DEP)
	@echo "\n=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

# Needs to be here! 
$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f *.depend
