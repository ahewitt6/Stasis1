# PRyMordial

A Python package for precise computations of BBN within and beyond the Standard Model.

[PRyM in a Nutshell](https://github.com/vallima/PRyMordial/files/11711841/PRyM.pdf)

To run a quick example, open the terminal and once in PRyMordial folder just type: 

python runPRyM_julia.py

Examples on the usage of PRyMordial are provided in the Jupyter notebooks PRyMdemoSM and PRyMdemoNP.

A dedicated numerical table for standard cosmological analyses involving BBN likelihoods is also in the repo.
For more details, see PRyM_Yp_DH_cosmoMC_2023.dat

Dependencies:
-------------
- NumPy (mandatory) – pip install numpy
- SciPy (mandatory) – pip install scipy
- Numba (recommended) – pip install numba
- Numdifftools (recommended) – pip install numdifftools
- Vegas (recommended) – pip install vegas
- PyJulia (optional) – pip install julia
- diffeqpy (optional) – pip install diffeqpy

The code can easily avoid dependencies on Numba and Numdifftools and it does not make use of Vegas by default.
The installation of these libraries still remains recommended for the best possible usage of the package.

The optional dependencies above require:
- the Julia programming language, https://julialang.org
- the open-source software for scientific machine learning, https://sciml.ai

Authors:
--------
See doc/CREDITS file for list of contributors to PRyMordial.

Availability:
-------------
See doc/COPYING and doc/LICENSE for licensing terms.

Documentation:
--------------
See [PRyMordial paper](https://arxiv.org/abs/2307.07061)!


