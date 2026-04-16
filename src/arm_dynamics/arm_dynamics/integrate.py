import numpy as np
def rk4_step(x,tau,dt,ddq_fn):
    """x=[q(4), dq(4)]"""
    def f(x_):
        q=x_[:4]
        dq=x_[4:]
        ddq=ddq_fn(q,dq,tau)
        return np.hstack([dq,ddq])
    k1 = f(x)
    k2 = f(x+0.5*dt*k1)
    k3 = f(x+0.5*dt*k2)
    k4 = f(x+dt*k3)
    return x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    