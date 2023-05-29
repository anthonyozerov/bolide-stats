This folder contains MATLAB code for obtaining the simulated latitudinal impact
distributions of asteroids with fixed velocities and meteor showers with fixed
radiants and velocities.

getdist.m and impact-test.m are adapted by Anthony Ozerov from code written by
Darrel Robertson for the 2021 paper "Latitude variation of flux and impact
angle of asteroid collisions with Earth and the Moon."

pvi2018.dat is a table representing the joint distribution of asteroid radiant
velocity and ecliptic latitude, computed using code written by Petr Pokorny and
modified by Darrel Robertson using the Granvik 2018 synthetic asteroid database.

The other .m files are part of a quaternion library written by Darrel Robertson.

All code here will run fine (albeit slower) under GNU Octave, and in fact this
is how it is called in impacts.py in the parent directory.
