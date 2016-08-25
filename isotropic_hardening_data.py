import numpy as np
#import Error_Plots as Plot

'number of points'
nup = 301
'number of examples'
m = 300
'noise range'
starty = 50
endy = 90

startx = 0.001
endx = 0.1

""" Neural Network parameter """
input_number = 7
output_number = 1
weight_1_1 = 7
weight_1_2 = 4
weight_2_1 =4
weight_2_2 = 4
weight_3_1 = 4
weight_3_2 = 1
bias_1 = 4
bias_2 = 4
bias_3 = 1 


"""
input: strain in [-]
output: stress in [MPa]
"""

'Material Parameter'
""" 
Youngs modulus E = 70*10^3 MPa
Shear modulus G = 26*10^3 MPa
bulk modulus K = 76*10^3 MPa
Lame-parameter: lambda = K - (2/3) G
                mu = G
yield stress Sig_y = 276 MPa
hardening parameter hard_para = 780 MPa
hardening exponent hard_exp = 0.17
"""
Youngs = 70.0*10**3
Shear = 26.0*10**3
Bulk = 76.0*10**3
Lambda = (Bulk - (2/3)*Shear)
mu = Shear
sigma_y = 276.0
hard_para = 780.0
hard_exp = 0.17


def Create_x_values():
    x = np.linspace(0,0.1,nup)
    return x

def Create_x_vector(noise_ratio):
    old_x = Create_x_values()
    x = np.zeros((nup,1))
    noise = np.random.uniform(0, noise_ratio, size=nup)
    for i in range(1,nup):
        x[i] = old_x[i]*100 + noise[i]
    return x

def Create_Function(noise_ratio):
    x = Create_x_values()
    elastic_plastic = np.random.randint(0, 3)
    end_elastic = 12 + elastic_plastic
    noise = np.random.uniform(0, noise_ratio, size=nup)
    y = np.zeros((nup,1))
    for i in range(1,end_elastic):
        y[i] = Youngs * x[i] + noise[i]
    for j in range(end_elastic,nup):
        y[j] = hard_para * (x[j] - (sigma_y-110)/Youngs)**hard_exp + noise[j] 
    return y

class Datas_Generator(object):
    
    def __init__(self,number_examples):
        self.number_examples = number_examples
    
    def Create_input_matrix(self,n_start,n_end):
        Noise_Ratio = np.random.uniform(n_start, n_end, size=self.number_examples)
        inputs = Create_x_vector(Noise_Ratio[i])
        zero_matrix = np.zeros((nup-1,1))
        Youngs_matrix = np.vstack((Youngs,zero_matrix)) 
        Shear_matrix = np.vstack((Shear,zero_matrix))
        Bulk_matrix = np.vstack((Bulk,zero_matrix))  
        sigma_y_matrix = np.vstack((sigma_y,zero_matrix))
        hard_para_matrix = np.vstack((hard_para,zero_matrix))
        hard_exp_matrix = np.vstack((hard_exp,zero_matrix))   
        input_matrix = np.column_stack((inputs,Youngs_matrix,Shear_matrix,Bulk_matrix,sigma_y_matrix,hard_para_matrix,hard_exp_matrix))
        return input_matrix
      
    def Create_output_matrix(self,n_start,n_end):
        Noise_Ratio = np.random.uniform(n_start, n_end, size=self.number_examples)
        output_matrix = Create_Function(Noise_Ratio[i])
        return output_matrix


output_matrixs=[]
input_matrixs=[]

for i in range(m):
    input_matrixs.append(Datas_Generator(m).Create_input_matrix(startx,endx))
    output_matrixs.append(Datas_Generator(m).Create_output_matrix(starty,endy))


total = m
training = 0.6*total
if (total-training) % 2 != 0:
    training = round(training)
cv = (total - training) / 2
test = (total - training) / 2
if cv != int(cv):
    training += 1
    cv = (total - training) / 2
    test = (total - training) / 2 
if (training+cv+test) != total:
    print 'number of seperated data is not right'
if training != int(training) or cv != int(cv) or test != int(test):
    print 'Error - some float number'


# 'raw plot (without noise)'
# xraw = Create_x_values()
# end_elastic = 12
# yraw = np.zeros((nup,1))
# for i in range(1,end_elastic):
#     yraw[i] = Youngs * xraw[i] 
# for j in range(end_elastic,nup):
#     #yraw[j] = (yraw[end_elastic]-hard_para * (xraw[end_elastic] - sigma_y/Youngs)**hard_exp) + hard_para * (xraw[j] - sigma_y/Youngs)**hard_exp 
#     yraw[j] = hard_para * (xraw[j] - (sigma_y-110)/Youngs)**hard_exp 
# 
# figure_raw = Plot.Einzel_Plot(xraw, yraw)


# 'Test Plots'
# test1y = output_matrixs[0]
# test2y = output_matrixs[50]
# 
# test1x = input_matrixs[0]
# x1 = test1x[:,0]
# test2x = input_matrixs[50]
# x2 = test2x[:,0]
# x_values = test1x[:,0]
# 
# figure1 = Plot.Doppel_Plot(x1, test1y[:,0], x2,test2y[:,0])
#figure2 = Plot.Einzel_Plot(x_values, test1y)



# 'saving in a .dat file'
# for i in range(m):
#     input_matrix = Datas_Generator(m).Create_input_matrix()
#     output_matrix = Datas_Generator(m).Create_output_matrix(start,end)
#     filenames = "data/{0}.dat".format(i)
#     with open('data/{0}.dat'.format(i),'wb') as f:
#         f.write('strain \t YoungsModulus \t ShearModulus \t BulkModulus \t YieldStress \t HardeningParameter \t HardeningExponent \t stress \n')
#         np.savetxt(f,np.transpose([input_matrix[:,0],input_matrix[:,1],input_matrix[:,2],input_matrix[:,3],input_matrix[:,4],input_matrix[:,5],input_matrix[:,6],output_matrix[:,0]])) 
#     input_matrixs.append(input_matrix)
#     output_matrixs.append(output_matrix)
      
