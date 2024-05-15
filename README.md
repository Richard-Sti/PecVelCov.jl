# Peculiar Velocities Covariance Matrix


## TODO
- [x] Write further speed-ups of the $C_{ij}$ elements and the covariance matrix calculation.
- [ ] Improve the grid sampling for more speed.
- [ ] Update the $P(k)$ interpolator to be in log-log and cubic.
- [ ] Write a hybrid covariance matrix calculation that uses interpolator for far-spaced elements and direct calculation for close-spaced elements.
- [ ] Add growth factor prefactors to the $C_{ij}$ calculation.
- [ ] Add a function that constructs the covariance matrix from a set of observations.
