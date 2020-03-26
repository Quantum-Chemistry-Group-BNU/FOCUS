#include "vmc.h"


// Ansatz |Psi>=tr(A[1]*A[2]*...*A[k])|n1n2...nk> ~ O(KD^2)
//
// 	  |Psi>=tr(A[p1]*...*A[pn])|p1...pn> ~ O(KD^2)
//	       =P_N*|FS-MPS>


// local energy: E[n] = <n|H|Psi>/<n|Psi>

// P[n] = |<n|Psi>|^2/<Psi|Psi>

// Gradients

// Optimize

