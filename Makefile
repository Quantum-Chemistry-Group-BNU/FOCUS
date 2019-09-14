USE_MPI = no
USE_INTEL = no #yes
EIGEN=/usr/local/Cellar/eigen/3.3.7/include/eigen3
BOOST=/usr/local

FLAGS = -std=c++11 -g  -O3 -I${EIGEN} -I${BOOST}/include 
DFLAGS = -std=c++11 -g -O3 -I${EIGEN} -I${BOOST}/include -DComplex

ifeq ($(USE_INTEL), yes) 
	FLAGS += -qopenmp
	DFLAGS += -qopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpiicpc
		CC = mpiicpc
		LFLAGS = -L${BOOST}/lib -lboost_serialization -lboost_mpi #-lrt
	else
		CXX = icpc
		CC = icpc
		LFLAGS = -L${BOOST}/lib -lboost_serialization #-lrt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
else
	FLAGS += -fopenmp
	DFLAGS +=  -fopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpicxx
		CC = mpicxx
		LFLAGS += -L${BOOST}/lib -lboost_serialization -lboost_mpi #-lrt
	else
		CXX = g++
		CC = gcc
		LFLAGS = -L${BOOST}/lib -lboost_serialization #-lrt
		FLAGS += -DSERIAL 
		DFLAGS += -DSERIAL 
	endif
endif


SRC_Matrix++ = main.cpp input.cpp integral.cpp

OBJ_Matrix++ += obj/main.o \
		obj/input.o \
		obj/integral.o

obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj_z/%.o: %.cpp  
	$(CXX) $(DFLAGS) $(OPT) -c $< -o $@

all: Matrix++

Matrix++ : $(OBJ_Matrix++) 
	$(CXX)   $(FLAGS) -o Matrix++ $(OBJ_Matrix++) $(LFLAGS)

clean :
	rm -f Matrix++;
	rm -f obj/*.o
