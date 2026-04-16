import numpy as np
def default_params():
    l=np.array([0.35,0.30,0.25,0.20],dtype=float)
    m=np.array([2.0,1.6,1.2,0.8],dtype=float)
    lc=0.5*l
    I=(1.0/12.0)*m*(l**2)#slender rods about COM
    g0=-9.81
    D=np.diag([4.00,3.00,2.00,1.6])
    tau_max=np.array([60.0,40.0,20.0,10.0],dtype=float)
    return dict (l=l,m=m,lc=lc,I=I,g0=g0,D=D,tau_max=tau_max)