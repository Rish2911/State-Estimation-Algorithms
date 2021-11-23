import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
V = np.array([[1.23, 1.38, 2.06, 2.47, 3.17]]).T

plt.scatter(I, V)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

## Batch Solution

H = np.ones((5, 2))
H[:, 0] = I.ravel()
print(H)
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The slope and offset parameters of the best-fit line (i.e., the resistance and offset) are [R, b]:')
print(x_ls[0, 0])
print(x_ls[1, 0])

# Plot line.
I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)
V_line = x_ls[0]*I_line + x_ls[1]

plt.scatter(I, V)
plt.plot(I_line, V_line)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

## Recursive Solution

# Initialize the 2x1 parameter vector x (i.e., x_0).
x_k = np.array([[4.6 , 0]]).T

#print(x_k)

#Initialize the 2x2 covaraince matrix (i.e. P_0). Off-diangonal elements should be zero.
P_k = np.array([[9 , 0],[0, 0.2]])
#print("P_k is", P_k)
# Our voltage measurement variance (denoted by R, don't confuse with resistance).
R_k = np.array([[0.0225]])

# Pre allocate space to save our estimates at every step.
num_meas = I.shape[0]
print("num_meas is", num_meas)
x_hist = np.zeros((num_meas + 1, 2)) 
P_hist = np.zeros((num_meas + 1, 2, 2))

x_hist[0] = x_k.T #what to do here
x_hist_t = np.zeros((2,1))
#print("x_hist is", x_hist)
P_hist[0] = P_k
Id = np.identity(2)

# Iterate over all the available measurements.
for k in range(num_meas):
    # Construct H_k (Jacobian).
    H_k = np.matrix(H[k]) #np.array does't work with .T  
    #print("the value of k is", k)
    #print("H_k trans is", H_k.T)
    #print("P_hist[k-1] is", P_hist[k-1])
    # Construct K_k (gain matrix).
    K_k = P_hist[k].dot(H_k.T)
    print("K_k is", K_k)              
    # Update our estimate.
    #print("new x_hist", k, "is", x_hist[k])
    x_hist_t[0][0] = x_hist[k][0]
    x_hist_t[1][0] = x_hist[k][1]
    print("new x_hist_t is", x_hist_t)
    x_k = x_hist_t + K_k.dot(V[k]-H_k.dot(x_hist_t))
    print("new x_k is", x_k)
    # Update our uncertainty (covariance)
    P_k = (Id - K_k.dot(H_k)).dot(P_hist[k]).dot((Id - K_k.dot(H_k)).T) + K_k.dot(R_k).dot(K_k.T) 
    print("new P_k is", P_k)   

    # Keep track of our history.
    P_hist[k + 1] = P_k
    x_hist[k + 1] = x_k.T
    
print('The slope and offset parameters of the best-fit line (i.e., the resistance and offset) are [R, b]:')
print(x_k)
print(x_k[0, 0])
print(x_k[1, 0])

plt.scatter(I, V, label='Data')
plt.plot(I_line, V_line, label='Batch Solution')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)

I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)

for k in range(num_meas):
    V_line = x_hist[k, 0]*I_line + x_hist[k, 1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.legend()
plt.show()


#signbed by Rishabh Singh