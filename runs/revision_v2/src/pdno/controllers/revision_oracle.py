"""Revision-only batched reference candidate rollouts; same discrete reference schemes."""
import numpy as np
from scipy.linalg import solve_banded
from pdno.data.generate import _periodic_actuators,_dirichlet_actuators
from pdno.data.teacher_queries import restrict_field
from pdno.controllers.linear import estimate_modes,_basis

def future_candidates(initial,pde,material,actions,ticks=8):
    n=len(initial);a=np.asarray(actions,dtype=np.float64);q=np.broadcast_to(np.asarray(initial,dtype=np.float64),(len(a),n)).copy();out=[]
    if pde=='burgers':
        force=a@_periodic_actuators(n);dt=2.5e-4;k=2*np.pi*np.fft.fftfreq(n,d=1/n);mul=np.exp(-float(material[0])*k*k*dt/2);dk=1j*k;dk[np.abs(np.fft.fftfreq(n)*n)>n/3]=0
        def rhs(v):return -np.fft.ifft(dk*np.fft.fft(.5*v*v,axis=-1),axis=-1).real+force
        for t in range(ticks):
            for _ in range(80):
                q=np.fft.ifft(mul*np.fft.fft(q,axis=-1),axis=-1).real
                k1=rhs(q);k2=rhs(q+.5*dt*k1);k3=rhs(q+.5*dt*k2);k4=rhs(q+dt*k3)
                q=q+dt/6*(k1+2*k2+2*k3+k4);q=np.fft.ifft(mul*np.fft.fft(q,axis=-1),axis=-1).real
            out.append(q[:,::2].astype(np.float32))
    else:
        dt=.02;dx=1/(n+1);r=float(material[0])*dt/(2*dx*dx);c=float(material[1])*dt/2;force=a@_dirichlet_actuators(n)
        ab=np.zeros((3,n));ab[0,1:]=-r;ab[1]=1+2*r+c;ab[2,:-1]=-r
        for t in range(ticks):
            b=(1-2*r-c)*q;b[:,1:]+=r*q[:,:-1];b[:,:-1]+=r*q[:,1:];b+=dt*force
            q=solve_banded((1,1),ab,b.T).T;out.append(np.stack([restrict_field(v,'heat') for v in q]))
    return np.stack(out,axis=1)

def observer_initial(obs,pde):
    x=np.arange(256)/256 if pde=='burgers' else np.arange(1,257)/257
    return _basis(pde,x,modes=7)@estimate_modes(obs,pde,modes=7)
