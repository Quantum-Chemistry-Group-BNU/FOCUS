#ifndef SPACE_H
#define SPACE_H

#include <iostream>
#include <vector>
#include <memory>
#include "onstate.h"
#include "integral.h"

namespace fock{

using onspace = std::vector<onstate>;
      
// print
void check_space(onspace& space);

// spinless case
onspace fci_space(const int k, const int n);

// k - number of spin orbitals 
onspace fci_space(const int k, const int na, const int nb);
      
// generate represenation of H in this space
//std::unique_ptr<double> genH();

}

#endif
