"""Revision positive thermal initial conditions, fixed before target generation."""
import numpy as np

def positive_heat_initial(rng,n):
    x=np.arange(1,n+1,dtype=float)/(n+1)
    v=sum(rng.normal()/k*np.sin(k*np.pi*x) for k in range(1,9))
    shape=np.sin(np.pi*x)*(.2+v*v)
    return shape*(rng.uniform(.1,.5)/shape.max())
