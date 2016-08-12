import numpy as np
#import tb_gosl as gosl
#import matplotlib.pyplot as plt

'number of points'
nup = 101

 
def Create_x_values():
    x = np.linspace(0,1,nup)
    return x

def Create_x_vector():
    old_x = Create_x_values()
    x = np.zeros((nup,1))
    for i in range(nup):
        x[i] = old_x[i]
    return x

def Create_Function_Noise1(noise_ratio):
    x = Create_x_values()
    y = np.zeros((nup,1))
    noise = np.random.uniform(0, noise_ratio, size=101)
    for i in range(nup):
        y[i] = x[i]**0.5 + noise[i]
    return y

def Create_Function_Noise2(noise_ratio):
    x = Create_x_values()
    y = np.zeros((nup,1))
    noise = np.random.uniform(0, noise_ratio, size=101)
    for i in range(nup):
        y[i] = 1.0 / ( 1.0 + np.exp(-x[i])) + noise[i]
    return y

def Create_Function_Noise3(noise_ratio):
    x = Create_x_values()
    y = np.zeros((nup,1))
    noise = np.random.uniform(0, noise_ratio, size=101)
    for i in range(nup):
        y[i] = x[i] + (np.exp(x[i]))**0.5 * np.tan(500/(x[i]+1)) + noise[i]
    return y

# 
#         
# y1=Create_Function_Noise(0.1)
# y2=Create_Function_Noise(0.1)
#  
# ' Test '
# plt.plot(y1,'--ro')
# plt.plot(y2,'--bo')
# plt.show() 