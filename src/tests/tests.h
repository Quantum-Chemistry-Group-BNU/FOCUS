#ifndef TESTS_H
#define TESTS_H

namespace tests{

// --- core ---
int test_tools();
int test_matrix();
int test_linalg();
int test_onstate();
int test_onspace();
int test_integral();
int test_hamiltonian();
int test_dvdson();
int test_simpleci();

// --- sci ---
int test_fci();
int test_sci();

// --- Comb ---
int test_tns();
int test_ctns();

}

#endif
