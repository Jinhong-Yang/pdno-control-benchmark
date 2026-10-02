"""Fixed-grid inference cache used only by revision latency experiments."""
import math
import numpy as np
import torch
from torch import nn
from pdno.models.operators import ActionFactorizedOperator,spatial_basis,trunk_coordinates,polynomial_action_features
from pdno.data.teacher_queries import feasible_candidates
from pdno.controllers.actions import project_box_slew

class CachedOperator(nn.Module):
    def __init__(self,model,x,tau):
        super().__init__();self.model=model;self.pde=model.pde;self.factorized=isinstance(model,ActionFactorizedOperator)
        with torch.no_grad():
            self.register_buffer('cached_trunk',model.trunk(trunk_coordinates(self.pde,x,tau)).detach().clone())
            self.register_buffer('cached_basis',spatial_basis(self.pde,x).detach().clone())
        self.grid_shape=(tuple(x.shape),tuple(tau.shape))
    def forward(self,obs,actions,x,tau):
        if (tuple(x.shape),tuple(tau.shape))!=self.grid_shape:raise ValueError('Cache is only valid for the fixed evaluation grid')
        m=self.model;context=m.encoder(obs);qc=m.initial_head(context);batch,candidates,_=actions.shape
        q0=torch.einsum('bq,qn->bn',qc,self.cached_basis)
        low,high=(-1.,1.) if self.pde=='burgers' else (0.,1.)
        phi=polynomial_action_features(actions,low,high)
        if self.factorized:
            coeff=m.response_branch(torch.cat((context,qc),dim=-1)).view(-1,m.p_terms,m.rank)
            response=torch.einsum('bpr,hnr->bphn',coeff,self.cached_trunk)
            delta=torch.einsum('bkp,bphn->bkhn',phi,response)
        else:
            rep=torch.cat((context,qc),dim=-1)[:,None,:].expand(-1,candidates,-1)
            coeff=m.action_branch(torch.cat((rep,actions),dim=-1).reshape(batch*candidates,-1)).view(batch,candidates,m.p_terms,m.rank)
            response=torch.einsum('bkpr,hnr->bkphn',coeff,self.cached_trunk)
            delta=torch.einsum('bkp,bkphn->bkhn',phi,response)
        if self.pde=='heat':delta=delta*(x*(1-x)).reshape(1,1,1,-1)
        return q0[:,None,None,:]+tau.reshape(1,1,-1,1)*delta

def expanded_candidates(pde,previous,nominal,K):
    if K==10:return feasible_candidates(pde,previous,nominal)
    if K not in [25,50,100,200]:raise ValueError('Unspecified candidate count')
    lo,hi,delta=(-1.,1.,.15) if pde=='burgers' else (0.,1.,.1)
    n=math.ceil(math.sqrt(K-2));v=np.linspace(-delta,delta,n);grid=np.array([[a,b] for a in v for b in v]);ix=np.linspace(0,n*n-1,K-2,dtype=int)
    offsets=grid[ix];out=[project_box_slew(previous+d,previous,lo,hi,delta) for d in offsets]
    out += [project_box_slew(previous,previous,lo,hi,delta),project_box_slew(nominal,previous,lo,hi,delta)]
    return np.array(out,dtype=np.float32)
