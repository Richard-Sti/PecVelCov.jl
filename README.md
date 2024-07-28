# Peculiar Velocities Covariance Matrix


## About

Code to calculate the covariance matrix of line-of-sight peculiar velocities assuming LCDM linear gravity, as described in [1]. The covariance matrix is:

```math
C_{ij}
=
\frac{1}{2\pi^2} \frac{\mathrm{d}D}{\mathrm{d}\tau(r_i)} \frac{\mathrm{d}D}{\mathrm{d}\tau(r_j)} \int{\rm d}k\,P(k) \sum_{\ell}^\infty (2\ell+1)j_{\ell}'(kr_i)j_{\ell}'(kr_j)P_{\ell}(\hat{\bf r}_i\cdot\hat{\bf r}_j).
```

## Installation


First, clone the repository:
```bash
git clone git@github.com:Richard-Sti/PecVelCov.jl.git
```

To install `GSHEIntegrator.jl`:
```bash
cd PeVelCov.jl
```
Then, open a Julia REPL and type:
```julia
julia
```
Then, in the Julia REPL, enter the package manager mode by pressing ] and type:
```bash
dev .
precompile
```

This will install the package in development along with its dependencies. To test the installation, you may run in the Julia REPL:
```julia
using PecVelCov
```

## License and Citation
If you use or find useful any of the code in this repository, please cite [1].

```
Copyright (C) 2024 Richard Stiskalek
This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
```

## Contributors
- Richard Stiskalek (University of Oxford)


## Examples
..
