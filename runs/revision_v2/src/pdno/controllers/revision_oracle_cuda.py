"""FP64 CUDA implementation of the unchanged periodic reference time step."""
import numpy as np
import torch
from pdno.data.generate import _periodic_actuators

class BurgersCandidateCUDA:
    def __init__(self,nu,K=10,n=256):
        self.B=len(nu);self.K=K;self.n=n;dev='cuda';self.dt=2.5e-4
        self.q=torch.zeros((self.B,K,n),device=dev,dtype=torch.float64)
        self.force=torch.zeros_like(self.q)
        k=2*np.pi*np.fft.fftfreq(n,d=1/n);mode=np.fft.fftfreq(n)*n
        dk=1j*k;dk[np.abs(mode)>n/3]=0
        self.dk=torch.tensor(dk,device=dev,dtype=torch.complex128)
        self.mul=torch.exp(-torch.tensor(nu,device=dev,dtype=torch.float64)[:,None,None]*torch.tensor(k*k,device=dev)*self.dt/2)
        self.basis=torch.tensor(_periodic_actuators(n),device=dev,dtype=torch.float64)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._outer_step();self._outer_step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self._outer_step()
    def rhs(self,q):return -torch.fft.ifft(self.dk*torch.fft.fft(.5*q*q,dim=-1),dim=-1).real+self.force
    def _outer_step(self):
        q=self.q
        for _ in range(80):
            q=torch.fft.ifft(self.mul*torch.fft.fft(q,dim=-1),dim=-1).real
            k1=self.rhs(q);k2=self.rhs(q+.5*self.dt*k1);k3=self.rhs(q+.5*self.dt*k2);k4=self.rhs(q+self.dt*k3)
            q=q+self.dt/6*(k1+2*k2+2*k3+k4)
            q=torch.fft.ifft(self.mul*torch.fft.fft(q,dim=-1),dim=-1).real
        self.q.copy_(q)
    @torch.no_grad()
    def __call__(self,initial,actions):
        q=torch.as_tensor(initial,device='cuda',dtype=torch.float64);a=torch.as_tensor(actions,device='cuda',dtype=torch.float64)
        self.q.copy_(q[:,None,:].expand(-1,self.K,-1));self.force.copy_(a@self.basis)
        outputs=[]
        for _ in range(8):
            self.graph.replay();outputs.append(self.q[:,:,::2].float().clone())
        return torch.stack(outputs,dim=2)

    @torch.no_grad()
    def advance_one_tick(self,initial,actions,extra_force=None):
        if self.K!=1:raise ValueError("Truth advance requires K=1")
        q=torch.as_tensor(initial,device='cuda',dtype=torch.float64);a=torch.as_tensor(actions,device='cuda',dtype=torch.float64)
        self.q.copy_(q[:,None,:]);self.force.copy_(a[:,None,:]@self.basis)
        if extra_force is not None:self.force.add_(torch.as_tensor(extra_force,device='cuda',dtype=torch.float64)[:,None,:])
        self.graph.replay()
        return self.q[:,0].float().cpu().numpy()
