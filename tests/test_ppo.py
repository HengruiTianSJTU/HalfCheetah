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


def test_gae_bootstraps_rollout_tail_without_episode_end():
    rewards=np.array([[1.],[2.]],np.float32)
    zeros=np.zeros_like(rewards)
    adv,returns=gae(rewards,zeros,np.array([[0.],[4.]],np.float32),zeros,gamma=.5,lam=1.)
    np.testing.assert_allclose(adv[:,0],[3.,4.])
    np.testing.assert_allclose(returns,adv)


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


def test_checkpoint_restores_policy_and_normalizer(tmp_path):
    model=ActorCritic();rms=RunningMeanStd((17,))
    rms.update(np.random.default_rng(42).normal(size=(32,17)))
    obs=np.random.default_rng(43).normal(size=(3,17))
    expected=model.actor(torch.as_tensor(rms.normalize(obs))).detach()
    path=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),obs_rms=rms.state_dict()),path)
    state=torch.load(path,weights_only=False)
    restored=ActorCritic();restored.load_state_dict(state['model'])
    stats=RunningMeanStd((17,));stats.load_state_dict(state['obs_rms'])
    torch.testing.assert_close(restored.actor(torch.as_tensor(stats.normalize(obs))),expected)
