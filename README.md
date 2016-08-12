# Unit_NeuralNetwork

Main file: 'Unit_NN_Datafiles_Backprop'

Main file contains/uses 'Data_Set_Generator' and 'Error_Plots'

'Error_Plots': plots the training and cross-validation error

'Data_Set_Generator': stores all data into input and output matrices 
  input parameters:
              number of examples 'm'
              noise range values from 'start' to 'end'
  input matrix is named 'input_matrix' and contains: 101 x values between 0 and 1
  output matrix is named 'output_matrix' and contains: between 1 and 3 diffrent y(x) functions
  y(x) is created with 'Data_Generator'

'Data_Generator' contains: Creates functions y(x) with noise
