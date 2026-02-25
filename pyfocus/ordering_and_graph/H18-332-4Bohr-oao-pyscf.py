from pyscf.scf import hf
from pyscf import ao2mo, gto, scf, fci, cc, ci, lo

atom: str = ""
for k in range(2):
    for j in range(3):
        for i in range(3):
            x = i * 4.0
            y = j * 4.0
            z = k * 4.0
            atom += f"H, {x:.2f}, {y:.2f}, {z:.2f} ;\n"
mol = gto.Mole(unit='bohr',atom=atom, verbose=5, basis="sto-3g", symmetry=False)
localized_orb=True,
localized_method="lowdin"

mol.build()
sorb = mol.nao * 2
nele = mol.nelectron
mf = scf.RHF(mol)
mf.init_guess = "atom"
mf.level_shift = 0.0
mf.max_cycle = 200
mf.conv_tol = 1.0e-14
e_hf = mf.scf()
print('e_hf:',e_hf)

# Integral interface
# iface = Iface()
# iface.mol = mol
# iface.mf = mf
# iface.nfrozen = 0

# Localized orbitals
coeff_lo = lo.orth_ao(mf, localized_method)
mo_coeff = coeff_lo

# info = iface.get_integral(mo_coeff)
# Iface.dump(info, fname=integral_file)
