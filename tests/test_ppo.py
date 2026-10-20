import numpy as np
import torch
from halfcheetah.ppo import ActorCritic, RunningMeanStd, clipped_objective, gae


def test_gae_time_limit_bootstraps_but_stops_recursion():
    r=np.array([[1.],[100.]],np.float32)
    v=np.array([[2.],[3.]],np.float32)
    nv=np.array([[5.],[0.]],np.float32)
    adv,ret=gae(r,v,nv,np.array([[1.],[1.]]),gamma=.9,lam=.95)
    np.testing.assert_allclose(adv[:,0],[3.5,97.])
    np.testing.assert_allclose(ret[:,0],[5.5,100.])


def test_clipping_both_advantage_signs():
    ratio=torch.tensor([1.5,.5,1.5,.5])
    adv=torch.tensor([1.,1.,-1.,-1.])
    torch.testing.assert_close(clipped_objective(ratio.log(),adv,.2),torch.tensor([1.2,.5,-1.5,-.8]))


def test_running_statistics_batch_equivalence():
    x=np.arange(60).reshape(20,3)
    one,two=RunningMeanStd((3,)),RunningMeanStd((3,))
    one.update(x);two.update(x[:7]);two.update(x[7:])
    np.testing.assert_allclose(one.mean,two.mean)
    np.testing.assert_allclose(one.var,two.var)


def test_actor_log_probability_and_gradient():
    model=ActorCritic()
    obs=torch.randn(8,17)
    action,old_logp,_,value=model.policy(obs)
    _,new_logp,_,_=model.policy(obs,action)
    torch.testing.assert_close(old_logp,new_logp)
    assert action.shape==(8,6) and value.shape==(8,)
    (-new_logp.mean()+value.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
