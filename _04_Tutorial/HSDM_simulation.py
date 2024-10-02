import numpy as np

def simulate_HSDM_2D(a, mu, ndt, sigma=1, dt=0.001):
    x = np.zeros(mu.shape)
    
    rt = 0
    
    while np.linalg.norm(x, 2) < a(rt):
        x += mu*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1, mu.shape)
        rt += dt
    
    theta = np.arctan2(x[1], x[0])   
    
    return ndt+rt, theta

def simulate_HSDM_3D(a, mu, ndt, sigma=1, dt=0.001):
    x = np.zeros(mu.shape)
    
    rt = 0
    
    while np.linalg.norm(x, 2) < a(rt):
        x += mu*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1, mu.shape)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(x[2], x[1])   
    
    return rt+ndt, (theta1, theta2)

def simulate_HSDM_4D(a, mu, ndt, sigma=1, dt=0.001):
    x = np.zeros(mu.shape)
    
    rt = 0
    
    while np.linalg.norm(x, 2) < a(rt):
        x += mu*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1, mu.shape)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2), x[1])
    theta3 = np.arctan2(x[3], x[2])
    
    
    return rt+ndt, (theta1, theta2, theta3)