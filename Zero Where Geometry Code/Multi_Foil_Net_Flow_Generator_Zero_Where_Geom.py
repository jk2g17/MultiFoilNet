import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.utils
import torch.optim as optim
from torch.linalg import vector_norm

from Multi_Foil_Net import MultiFoilNet, weights_init
import Data_Loader_Local_L1_CLF
import torch.nn.functional as F
from Utility_Functions import DeNormaliser
import timeit

######## Settings ########

# Batch size.
batch_size = 1
# Channel exponent to control the network size.
expo = 5
# Do you want to add dropout to the model? If yes, set to non-zero value.
dropout    = 0.
# Root directory of Samples.
Sample_Location = "D:/IP/01 - Training Data/"
# List of required samples for Test.
Test_Sample_List = ["Study_2_H_C"]#["Sample_12_1","Sample_12_2","Sample_12_3","Sample_12_4","Sample_12_5"]
# Load the required model.
doLoad = "D:/IP/00 - Training Area/02 - Models/00 - Full Training Set/Varied LRs - Epochs=15 - Exponent=5/Epochs=15 - LR=0_00001 - Exponent=5 - Zero Where Geom/Epoch_15.pt"
# Define save location.
SaveArea = "NN_Zeros_Study_2_H_C/"
# Custom loss function name.
CLF_Name = "CLF_Linear_Ramp_Step=0.2"
# BC name.
BC_Name = "Pressure_Neumann_BC"
# Local L1 Definition.
L1Def = "/Local_L1_5_Cells/"

##########################

######## Set up Test ID dictionary ########

k = 0
Test_Sample_Dictionary = {}
Test_ID_Dictionary = {}

# for i in range(len(Test_Sample_List)):
#     # This line finds the numbers in the string name.
#     ID_Index_Values = [int(s) for s in Test_Sample_List[i].split("_") if s.isdigit()]
#     for j in range(1,1001):
#         # This line calculates the relevant ID for the given dataset. Uses logic from the sample name.
#         ID = (((int(ID_Index_Values[0]) - 1) * 5000) + ((int(ID_Index_Values[1]) - 1) * 1000)) + j
#         # Create dictionary from this, where for a given index the sample can be found.
#         Test_Sample_Dictionary[k] = Test_Sample_List[i]
#         Test_ID_Dictionary[k] = ID
#         k += 1

for i in range(len(Test_Sample_List)):
    # This line finds the numbers in the string name.
    for j in range(1,37):
        # This line calculates the relevant ID for the given dataset. Uses logic from the sample name.
        ID = j
        # Create dictionary from this, where for a given index the sample can be found.
        Test_Sample_Dictionary[k] = Test_Sample_List[i]
        Test_ID_Dictionary[k] = ID
        k += 1        

##########################

######## Initialise Neural Network ########

# Call MultiFoilNet from respective file.
NeuralNet = MultiFoilNet(channelExponent=expo, dropout=dropout)
# Set model parameters.
model_parameters = filter(lambda p: p.requires_grad, NeuralNet.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized MultiFoilNet with {} trainable parameters. ".format(params))

# Apply weights via the function called from Multi_Foil_Net.
NeuralNet.apply(weights_init)
# If there is an existing model to load, do so via the following code.
if len(doLoad)>0:
    NeuralNet.load_state_dict(torch.load(doLoad))
    print("Loaded model {}.".format(doLoad))
# Set the neural network as a CUDA object.
NeuralNet.cuda()

##########################

######## Initialise dataset and DataLoader ########

# Call the dataset object made in Data_Loader.
MultiFoil_Dataset = Data_Loader_Local_L1_CLF.MultiFoilData_CLF(Sample_Location,Test_Sample_Dictionary,Test_ID_Dictionary,CLF_Name,BC_Name,L1Def)
# Initialise the DataLoader object for lazy loading.
Test_Loader = DataLoader(MultiFoil_Dataset, batch_size=batch_size, shuffle=False)

##########################

######## Initialise L1 Criterion, along with training objects ########

# Instantiate L1Loss function.
criterionL1 = nn.L1Loss(reduction='none')
# Set as CUDA object.
criterionL1.cuda()
# Initialise Training Objects.
targets = Variable(torch.FloatTensor(batch_size, 3, 512, 512))
inputs  = Variable(torch.FloatTensor(batch_size, 1, 512, 512))
targets = targets.cuda()
inputs  = inputs.cuda()
output_data = np.zeros([batch_size, 3, 512, 512])

##########################

######## Evaluate network against Test data ########

# Inititalise Neural Network object to train and initialise L1 parameter.    
NeuralNet.eval()
L1val_accum = 0.0
save_iter = 0

Test_Len = len(Test_Loader)*batch_size

Test_Batch_L1 = np.zeros(len(Test_Loader))
Test_Batch_Cp_L1 = np.zeros(len(Test_Loader))
Test_Batch_X_Vel_L1 = np.zeros(len(Test_Loader))
Test_Batch_Y_Vel_L1 = np.zeros(len(Test_Loader))

f_Test_Batch_L1 = np.zeros(len(Test_Loader))
f_Test_Batch_Cp_L1 = np.zeros(len(Test_Loader))
f_Test_Batch_X_Vel_L1 = np.zeros(len(Test_Loader))
f_Test_Batch_Y_Vel_L1 = np.zeros(len(Test_Loader))

Test_Batch_MAPE = np.zeros(len(Test_Loader))
Test_Batch_Cp_MAPE = np.zeros(len(Test_Loader))
Test_Batch_X_Vel_MAPE = np.zeros(len(Test_Loader))
Test_Batch_Y_Vel_MAPE = np.zeros(len(Test_Loader))

Local_Test_Batch_Cp_MAPE = np.zeros(len(Test_Loader))
Local_Test_Batch_X_Vel_MAPE = np.zeros(len(Test_Loader))
Local_Test_Batch_Y_Vel_MAPE = np.zeros(len(Test_Loader))

Test_Batch_Cp_Inf_Norm = np.zeros(Test_Len)
Test_Batch_Cp_Inf_Norm_X = np.zeros(Test_Len)
Test_Batch_Cp_Inf_Norm_Y = np.zeros(Test_Len)

Test_Batch_X_Vel_Inf_Norm = np.zeros(Test_Len)
Test_Batch_X_Vel_Inf_Norm_X = np.zeros(Test_Len)
Test_Batch_X_Vel_Inf_Norm_Y = np.zeros(Test_Len)

Test_Batch_Y_Vel_Inf_Norm = np.zeros(Test_Len)
Test_Batch_Y_Vel_Inf_Norm_X = np.zeros(Test_Len)
Test_Batch_Y_Vel_Inf_Norm_Y = np.zeros(Test_Len)

timer = 0

for i,Test_Data in enumerate(Test_Loader):
    inputs_cpu, targets_cpu, CLF_cpu, L1Def_cpu = Test_Data
    targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
    inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
    targets.data.resize_as_(targets_cpu).copy_(targets_cpu)
    
    tic=timeit.default_timer()
    outputs = NeuralNet(inputs)
    toc=timeit.default_timer()

    inputs_cpu = inputs_cpu.cpu()

    lossL1 = criterionL1(outputs, targets).cpu()
    lossL1[:,0,:,:] = lossL1[:,0,:,:] * inputs_cpu[:,0,:,:]
    lossL1[:,1,:,:] = lossL1[:,1,:,:] * inputs_cpu[:,0,:,:]
    lossL1[:,2,:,:] = lossL1[:,2,:,:] * inputs_cpu[:,0,:,:]
    lossL1 = torch.mean(lossL1[lossL1 != 0])
    
    lossL1_Cp = criterionL1(outputs[:,0,:,:],targets[:,0,:,:]).cpu()
    lossL1_Cp = lossL1_Cp * inputs_cpu[:,0,:,:]   
    lossL1_Cp = torch.mean(lossL1_Cp[lossL1_Cp != 0])
    
    lossL1_X_Vel = criterionL1(outputs[:,1,:,:],targets[:,1,:,:]).cpu()
    lossL1_X_Vel = lossL1_X_Vel * inputs_cpu[:,0,:,:]  
    lossL1_X_Vel = torch.mean(lossL1_X_Vel[lossL1_X_Vel != 0])
    
    lossL1_Y_Vel = criterionL1(outputs[:,2,:,:],targets[:,2,:,:]).cpu()
    lossL1_Y_Vel = lossL1_Y_Vel * inputs_cpu[:,0,:,:]    
    lossL1_Y_Vel = torch.mean(lossL1_Y_Vel[lossL1_Y_Vel != 0])

    f_lossL1 = criterionL1(outputs, targets).cpu()
    f_lossL1[:,0,:,:] = f_lossL1[:,0,:,:] * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1[:,1,:,:] = f_lossL1[:,1,:,:] * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1[:,2,:,:] = f_lossL1[:,2,:,:] * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1 = torch.mean(f_lossL1[f_lossL1 != 0])
    
    f_lossL1_Cp = criterionL1(outputs[:,0,:,:], targets[:,0,:,:]).cpu()
    f_lossL1_Cp = f_lossL1_Cp * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1_Cp = torch.mean(f_lossL1_Cp[f_lossL1_Cp != 0])
    
    f_lossL1_X_Vel = criterionL1(outputs[:,1,:,:], targets[:,1,:,:]).cpu()
    f_lossL1_X_Vel = f_lossL1_X_Vel * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1_X_Vel = torch.mean(f_lossL1_X_Vel[f_lossL1_X_Vel != 0])
    
    f_lossL1_Y_Vel = criterionL1(outputs[:,2,:,:], targets[:,2,:,:]).cpu()
    f_lossL1_Y_Vel = f_lossL1_Y_Vel * L1Def_cpu * inputs_cpu[:,0,:,:]
    f_lossL1_Y_Vel = torch.mean(f_lossL1_Y_Vel[f_lossL1_Y_Vel != 0]) 

    lossL1 = lossL1.item()
    lossL1_Cp = lossL1_Cp.item()
    lossL1_X_Vel = lossL1_X_Vel.item()
    lossL1_Y_Vel = lossL1_Y_Vel.item()

    f_lossL1 = f_lossL1.item()
    f_lossL1_Cp = f_lossL1_Cp.item()
    f_lossL1_X_Vel = f_lossL1_X_Vel.item()
    f_lossL1_Y_Vel = f_lossL1_Y_Vel.item()

    L1val_accum += lossL1
   
    Test_Batch_L1[i] = lossL1
    Test_Batch_Cp_L1[i] = lossL1_Cp
    Test_Batch_X_Vel_L1[i] = lossL1_X_Vel
    Test_Batch_Y_Vel_L1[i] = lossL1_Y_Vel

    f_Test_Batch_L1[i] = f_lossL1
    f_Test_Batch_Cp_L1[i] = f_lossL1_Cp
    f_Test_Batch_X_Vel_L1[i] = f_lossL1_X_Vel
    f_Test_Batch_Y_Vel_L1[i] = f_lossL1_Y_Vel   
    
    for j in range(batch_size):
        # Calculate infinity norm and locations.
        Cp_Delta = (outputs[j,0,:,:] - targets[j,0,:,:]) * inputs[j,0,:,:]
        X_Vel_Delta = (outputs[j,1,:,:] - targets[j,1,:,:]) * inputs[j,0,:,:]
        Y_Vel_Delta = (outputs[j,2,:,:] - targets[j,2,:,:]) * inputs[j,0,:,:]
        
        Cp_Delta = Cp_Delta[3:302,3:473]
        X_Vel_Delta = X_Vel_Delta[3:302,3:473]
        Y_Vel_Delta = Y_Vel_Delta[3:302,3:473]
        
        Cp_Infty_Norm = vector_norm(Cp_Delta,float('inf'))
        X_Vel_Infty_Norm = vector_norm(X_Vel_Delta,float('inf'))
        Y_Vel_Infty_Norm = vector_norm(Y_Vel_Delta,float('inf'))
        
        Coord_Cp = torch.where(abs(Cp_Delta) == Cp_Infty_Norm)
        Coord_X_Vel = torch.where(abs(X_Vel_Delta) == X_Vel_Infty_Norm)
        Coord_Y_Vel = torch.where(abs(Y_Vel_Delta) == Y_Vel_Infty_Norm)
    
        Cp_X = int(Coord_Cp[0][0])
        Cp_Y = int(Coord_Cp[1][0])
        
        X_Vel_X = int(Coord_X_Vel[0][0])
        X_Vel_Y = int(Coord_X_Vel[1][0])
                     
        Y_Vel_X = int(Coord_Y_Vel[0][0])
        Y_Vel_Y = int(Coord_Y_Vel[1][0])
        
        idx = (i*batch_size) + j
        
        Test_Batch_Cp_Inf_Norm[idx] = Cp_Infty_Norm
        Test_Batch_Cp_Inf_Norm_X[idx] = Cp_X
        Test_Batch_Cp_Inf_Norm_Y[idx] = Cp_Y
    
        Test_Batch_X_Vel_Inf_Norm[idx] = X_Vel_Infty_Norm
        Test_Batch_X_Vel_Inf_Norm_X[idx] = X_Vel_X
        Test_Batch_X_Vel_Inf_Norm_Y[idx] = X_Vel_Y
        
        Test_Batch_Y_Vel_Inf_Norm[idx] = Y_Vel_Infty_Norm
        Test_Batch_Y_Vel_Inf_Norm_X[idx] = Y_Vel_X
        Test_Batch_Y_Vel_Inf_Norm_Y[idx] = Y_Vel_Y    

    outputs_CPU = outputs.cpu()
    outputs_CPU = outputs_CPU.detach().numpy() 
    targets_CPU = targets.cpu()
    targets_CPU = targets_CPU.detach().numpy()
    inputs_CPU = inputs.cpu()
    inputs_CPU = inputs_CPU.detach().numpy()      
    
    output_data = np.array(outputs_CPU)
    output_data = output_data[:,:,0:304,0:475]
    output_data = DeNormaliser(output_data)
    
    Target = np.array(targets_CPU)
    Target[:,0,:,:] = Target[:,0,:,:] * inputs_CPU[:,0,:,:]
    Target[:,1,:,:] = Target[:,1,:,:] * inputs_CPU[:,0,:,:]
    Target[:,2,:,:] = Target[:,2,:,:] * inputs_CPU[:,0,:,:]
    
    Target_Cp = np.array(Target[:,0,:,:])
    Target_X_Vel = np.array(Target[:,1,:,:])
    Target_Y_Vel = np.array(Target[:,2,:,:])

    Output = np.array(outputs_CPU)
    Output[:,0,:,:] = Output[:,0,:,:] * inputs_CPU[:,0,:,:]
    Output[:,1,:,:] = Output[:,1,:,:] * inputs_CPU[:,0,:,:]
    Output[:,2,:,:] = Output[:,2,:,:] * inputs_CPU[:,0,:,:]    
    
    Output_Cp = np.array(Output[:,0,:,:])
    Output_X_Vel = np.array(Output[:,1,:,:])
    Output_Y_Vel = np.array(Output[:,2,:,:])
     
    Target = Target[:,:,0:304,0:475]
    Target_Cp = Target_Cp[:,0:304,0:475]
    Target_X_Vel = Target_X_Vel[:,0:304,0:475]
    Target_Y_Vel = Target_Y_Vel[:,0:304,0:475]

    Output = Output[:,:,0:304,0:475]
    Output_Cp = Output_Cp[:,0:304,0:475]
    Output_X_Vel = Output_X_Vel[:,0:304,0:475]
    Output_Y_Vel = Output_Y_Vel[:,0:304,0:475]
    
    L1Def_cpu = L1Def_cpu[:,0:304,0:475].detach().numpy()
    
    MAPE = (np.sum(np.abs(Output - Target))/np.sum(np.abs(Target))) * 100
    Cp_MAPE = (np.sum(np.abs(Output_Cp - Target_Cp))/np.sum(np.abs(Target_Cp))) * 100
    X_Vel_MAPE = (np.sum(np.abs(Output_X_Vel - Target_X_Vel))/np.sum(np.abs(Target_X_Vel))) * 100
    Y_Vel_MAPE = (np.sum(np.abs(Output_Y_Vel - Target_Y_Vel))/np.sum(np.abs(Target_Y_Vel))) * 100
    
    Local_Cp_MAPE = (np.sum(np.abs((Output_Cp - Target_Cp) * L1Def_cpu))/np.sum(np.abs(Target_Cp * L1Def_cpu))) * 100
    Local_X_Vel_MAPE = (np.sum(np.abs((Output_X_Vel - Target_X_Vel) * L1Def_cpu))/np.sum(np.abs(Target_X_Vel * L1Def_cpu))) * 100
    Local_Y_Vel_MAPE = (np.sum(np.abs((Output_Y_Vel - Target_Y_Vel) * L1Def_cpu))/np.sum(np.abs(Target_Y_Vel * L1Def_cpu))) * 100
    
    Test_Batch_MAPE[i] = MAPE
    Test_Batch_Cp_MAPE[i] = Cp_MAPE
    Test_Batch_X_Vel_MAPE[i] = X_Vel_MAPE
    Test_Batch_Y_Vel_MAPE[i] = Y_Vel_MAPE
    
    Local_Test_Batch_Cp_MAPE[i] = Local_Cp_MAPE
    Local_Test_Batch_X_Vel_MAPE[i] = Local_X_Vel_MAPE
    Local_Test_Batch_Y_Vel_MAPE[i] = Local_Y_Vel_MAPE
    
    for j in range(batch_size):
        savenameCp = SaveArea + "Cp/Cp_" + str(Test_ID_Dictionary[save_iter]) + ".npy"
        savenameXVel = SaveArea + "X_Vel/X_Vel_" + str(Test_ID_Dictionary[save_iter]) + ".npy"
        savenameYVel = SaveArea + "Y_Vel/Y_Vel_" + str(Test_ID_Dictionary[save_iter]) + ".npy"
        np.save(savenameCp,output_data[j,0,:,:])
        np.save(savenameXVel,output_data[j,1,:,:])
        np.save(savenameYVel,output_data[j,2,:,:])
        save_iter += 1
    
    timer += toc - tic
    
    print("Batch: {}, L1: {}".format(i+1,lossL1))
    
# Write L1 norm values and locations to file.
Test_L1_Log = "Total L1,Cp L1,X Velocity L1,Y Velocity L1,Total Local L1,Cp Local L1,X Velocity Local L1,Y Velocity Local L1\n"
    
for p in range(len(Test_Loader)):
    Test_L1_Log += str(Test_Batch_L1[p]) + "," + str(Test_Batch_Cp_L1[p]) + "," + str(Test_Batch_X_Vel_L1[p]) + "," + str(Test_Batch_Y_Vel_L1[p]) + "," + str(f_Test_Batch_L1[p]) + "," + str(f_Test_Batch_Cp_L1[p]) + "," + str(f_Test_Batch_X_Vel_L1[p]) + "," + str(f_Test_Batch_Y_Vel_L1[p]) + "\n"

m = open(SaveArea+"L1_Norms.csv", "w")
m.write(Test_L1_Log)
m.close()    

# Write infinity norm values and locations to file.
Test_Infty_Log = "Cp Infinity Norm,Cp X Coordinate,Cp Y Coordinate,X Velocity Infinity Norm,X Velocity X Coordinate,X Velocity Y Coordinate,Y Velocity Infinity Norm,Y Velocity X Coordinate,Y Velocity Y Coordinate\n"
    
for k in range(Test_Len):
    Test_Infty_Log += str(Test_Batch_Cp_Inf_Norm[k]) + "," + str(Test_Batch_Cp_Inf_Norm_X[k]) + "," + str(Test_Batch_Cp_Inf_Norm_Y[k]) + "," + str(Test_Batch_X_Vel_Inf_Norm[k]) + "," + str(Test_Batch_X_Vel_Inf_Norm_X[k]) + "," + str(Test_Batch_X_Vel_Inf_Norm_Y[k]) + "," + str(Test_Batch_Y_Vel_Inf_Norm[k]) + "," + str(Test_Batch_Y_Vel_Inf_Norm_X[k]) + "," + str(Test_Batch_Y_Vel_Inf_Norm_Y[k]) + "\n"

m = open(SaveArea+"Infinity_Norms.csv", "w")
m.write(Test_Infty_Log)
m.close()

# Write MAPE values and locations to file.
Test_MAPE_Log = "Total MAPE,Cp MAPE,X Velocity MAPE,Y Velocity MAPE,Local Cp MAPE,Local X Velocity MAPE,Local Y Velocity MAPE\n"
    
for n in range(len(Test_Loader)):
    Test_MAPE_Log += str(Test_Batch_MAPE[n]) + "," + str(Test_Batch_Cp_MAPE[n]) + "," + str(Test_Batch_X_Vel_MAPE[n]) + "," + str(Test_Batch_Y_Vel_MAPE[n]) + "," + str(Local_Test_Batch_Cp_MAPE[n]) + "," + str(Local_Test_Batch_X_Vel_MAPE[n]) + "," + str(Local_Test_Batch_Y_Vel_MAPE[n]) + "\n"

m = open(SaveArea+"MAPE.csv", "w")
m.write(Test_MAPE_Log)
m.close()  

print("The DNN took {} seconds to evalate this dataset.".format(timer))