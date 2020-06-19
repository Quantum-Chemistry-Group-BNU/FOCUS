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

/*
int test_rdm();
int test_sci();
int test_pt2();

// --- Comb ---
int test_comb();

// --- experiemental ---
int test_vmc();
int test_ordering();
int test_proj();
int test_ras();

*/

}

#endif
