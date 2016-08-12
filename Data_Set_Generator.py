import Data_Generator as data
import numpy as np
#import matplotlib.pyplot as plt

'number of examples'
m = 10000
'noise range'
start = 0.01
end = 0.05

class Datas_Generator(object):
    
    def __init__(self,number_examples):
        self.number_examples = number_examples
    
    def Create_input_matrix(self):
        input_matrix = data.Create_x_vector()
        return input_matrix
    
    def Create_output_matrix(self,n_start,n_end):
        Noise_Ratio = np.random.uniform(n_start, n_end, size=self.number_examples)
        #output_matrix = np.column_stack((data.Create_Function_Noise1(Noise_Ratio[i]),data.Create_Function_Noise2(Noise_Ratio[i]),data.Create_Function_Noise3(Noise_Ratio[i])))
        output_matrix = np.column_stack((data.Create_Function_Noise1(Noise_Ratio[i]),data.Create_Function_Noise2(Noise_Ratio[i])))
        return output_matrix
    
input_matrixs=[]
output_matrixs=[]

for i in range(m):
    input_matrixs.append(Datas_Generator(m).Create_input_matrix())
    output_matrixs.append(Datas_Generator(m).Create_output_matrix(start,end))

""" seperate all examples into training, cv, test data """
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




# ' Test '
# 
# y1=output_matrixs[0]
# y2=output_matrixs[5]
#  
# plt.plot(y1,'--ro')
# plt.plot(y2,'--bo')
# plt.show() 
