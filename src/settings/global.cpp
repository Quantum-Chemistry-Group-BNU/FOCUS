#include <iostream>
#include <cmath> // pow
#include "global.h"

using namespace std;

void global::license() {
   cout << "\n"+line_separator <<endl;
   cout << "Qubic: a platform for exploring the intersection of " << endl;
   cout << "	   Quantum Chemistry, Quantum Information, and Quantum Computation" << endl;
   cout << "Copyright (c) 2020 Zhendong Li" << endl;
   cout << "Author: Zhendong Li <zhendongli2008@gmail.com>" << endl;
   cout << line_separator+"\n" <<endl;
}

double global::memSize(size_t sz, const int dfac){
   return sz*dfac/pow(1024.0,2); // in MB
}
