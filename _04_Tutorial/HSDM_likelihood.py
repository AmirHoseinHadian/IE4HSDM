import numpy as np
from HSDM_fpt import fpt

def likelihood_2D(prms, RT, Response, collapsing=False, dt=0.1):
    
    if collapsing:
        a = lambda t: prms[0] - prms[1]*t
        a2 = lambda t: (a(t))**2
        da2 = lambda t: -2*prms[1] * a(t)
    
        ndt = prms[2]
        mu = np.array([prms[3], prms[4]])
        T_max = min(max(RT), prms[0]/prms[1])
    else:
        a = lambda t: prms[0]
        a2 = lambda t: prms[0]**2
        da2 = lambda t: 0
        
        ndt = prms[1]
        mu = np.array([prms[2], prms[3]])
        T_max = max(RT)
        
    pdf = fpt(a2, da2, mu.shape[0], 0.000001, dt=dt, T_max=T_max)
        
        
    log_lik = 0
    for i in range(len(RT)):
        rt, theta = RT[i], Response[i]
        if rt - ndt > 0.0001 and rt - ndt < T_max:
            mu_dot_x0 = mu[0]*np.cos(theta)
            mu_dot_x1 = mu[1]*np.sin(theta)
            
            if collapsing:
                term1 = a(rt - ndt) * (mu_dot_x0 + mu_dot_x1)
            else:
                term1 = prms[0] * (mu_dot_x0 + mu_dot_x1)
            
            term2 = 0.5 * np.linalg.norm(mu, 2)**2 * (rt - ndt)

            density = np.exp(term1 - term2) * pdf(rt - ndt)
            
            if 0.1**14 < density:
                log_lik += -np.log(density)
            else:
                log_lik += -np.log(0.1**14)
        else:
            log_lik += -np.log(0.1**14)
        
    return log_lik