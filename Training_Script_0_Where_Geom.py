import random, torch, torch.nn.utils, Data_Loader_Zero_Where_Geom, Utility_Functions, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.linalg import vector_norm
from Multi_Foil_Net import MultiFoilNet, weights_init
from os.path import exists
from os import listdir
from os import mkdir
import time


######## Settings ########

# Batch size.
batch_size = 10
# Learning Rate.
lrNN = 0.00001
# Do you want to decay the learning rate. True = yes.
decayLr = True
# Channel exponent to control the network size.
expo = 5
# Save model after every epoch?
SaveModelEpoch = False
# Save L1 text file after every epoch?
saveL1 = False
# Root directory of Samples.
Sample_Location = "D:/IP/01 - Training Data/"
# List of Required Samples for training.
Training_Sample_List = ["Sample_1_1","Sample_1_2","Sample_1_3","Sample_1_4","Sample_1_5","Sample_2_1","Sample_2_2","Sample_2_3","Sample_2_4","Sample_2_5",
                        "Sample_3_1","Sample_3_2","Sample_3_3","Sample_3_4","Sample_3_5","Sample_4_1","Sample_4_2","Sample_4_3","Sample_4_4","Sample_4_5",
                        "Sample_5_1","Sample_5_2","Sample_5_3","Sample_5_4","Sample_5_5","Sample_6_1","Sample_6_2","Sample_6_3","Sample_6_4","Sample_6_5",
                        "Sample_7_1","Sample_7_2","Sample_7_3","Sample_7_4","Sample_7_5","Sample_8_1","Sample_8_2","Sample_8_3","Sample_8_4","Sample_8_5",
                        "Sample_9_1","Sample_9_2","Sample_9_3","Sample_9_4","Sample_9_5","Sample_10_1","Sample_10_2","Sample_10_3","Sample_10_4","Sample_10_5"]
# List of Required Samples for validation.
Validation_Sample_List = ["Sample_11_1","Sample_11_2","Sample_11_3","Sample_11_4","Sample_11_5"]
# Define number of epochs
epochs = 15
# Apply gradient clipping?
Apply_Clip = False
# Gradient clip value.
clip = 1
# Do you want to add dropout to the model? If yes, set to non-zero value.
dropout = 0.
# Do you want to load a previously trained model? If so, what is the file location.
doLoad = ""
# The model save name.
SaveName = "D:/IP/00 - Training Area/02 - Models/00 - Full Training Set/Varied LRs - Epochs=15 - Exponent=5/Epochs=15 - LR=0_00001 - Exponent=5 - Zero Where Geom/"

##########################

######## Set up training and validation ID dictionary ########

k = 0
Training_Sample_Dictionary = {}
Training_ID_Dictionary = {}

m = 0
Validation_Sample_Dictionary = {}
Validation_ID_Dictionary = {}

for i in range(len(Training_Sample_List)):
    # This line finds the numbers in the string name.
    ID_Index_Values = [int(s) for s in Training_Sample_List[i].split("_") if s.isdigit()]
    for j in range(1,1001):
        # This line calculates the relevant ID for the given dataset. Uses logic from the sample name.
        ID = (((int(ID_Index_Values[0]) - 1) * 5000) + ((int(ID_Index_Values[1]) - 1) * 1000)) + j
        # Create dictionary from this, where for a given index the sample can be found.
        Training_Sample_Dictionary[k] = Training_Sample_List[i]
        Training_ID_Dictionary[k] = ID
        k += 1
        
for i in range(len(Validation_Sample_List)):
    # This line finds the numbers in the string name.
    ID_Index_Values = [int(s) for s in Validation_Sample_List[i].split("_") if s.isdigit()]
    for j in range(1,1001):
        # This line calculates the relevant ID for the given dataset. Uses logic from the sample name.
        ID = (((int(ID_Index_Values[0]) - 1) * 5000) + ((int(ID_Index_Values[1]) - 1) * 1000)) + j
        # Create dictionary from this, where for a given index the sample can be found.
        Validation_Sample_Dictionary[m] = Validation_Sample_List[i]
        Validation_ID_Dictionary[m] = ID
        m += 1
        
print("LR: {}".format(lrNN))
print("LR decay: {}".format(decayLr))
print("Epochs: {}".format(epochs))
print("Dropout: {}".format(dropout))

##########################

######## Check if model name already exists ########

if exists(SaveName):
    if len(listdir(SaveName)) != 0:
        print("Save directory already populated! Exiting code.")
        sys.exit()
    else:
        print("Specified save folder empty. Progressing with training.")
        mkdir(SaveName + "Training_Infinity_Norms/")
        mkdir(SaveName + "Validation_Infinity_Norms/")
else:
    mkdir(SaveName)
    mkdir(SaveName + "Training_Infinity_Norms/")
    mkdir(SaveName + "Validation_Infinity_Norms/")

##########################

######## Radom seed generation for Numpy and Torch ########

# Generate and print seed.
seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))

# Assign seed to respective modules.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

##########################

######## Initialise dataset and DataLoader ########

# Call the dataset object made in Data_Loader for training set.
MultiFoil_Training_Set = Data_Loader_Zero_Where_Geom.MultiFoilData(Sample_Location,Training_Sample_Dictionary,Training_ID_Dictionary)
# Initialise the DataLoader object for lazy loading.
Training_Loader = DataLoader(MultiFoil_Training_Set, batch_size=batch_size, shuffle=True)

# Call the dataset object made in Data_Loader for validation set.
MultiFoil_Validation_Set = Data_Loader_Zero_Where_Geom.MultiFoilData(Sample_Location,Validation_Sample_Dictionary,Validation_ID_Dictionary)
# Initialise the DataLoader object for lazy loading.
Validation_Loader = DataLoader(MultiFoil_Validation_Set, batch_size=batch_size, shuffle=True)

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

######## Initialise Optimizer function and L1 Criterion, along with training objects ########

# Instantiate L1 Loss function.
criterionL1 = nn.L1Loss(reduction='none')
# Set as CUDA object.
criterionL1.cuda()
# Initialise Adam Optimizer.
optimizerG = optim.Adam(NeuralNet.parameters(), lr=lrNN, betas=(0.5, 0.999), weight_decay=0.0)

##########################

# Instantiate the objects used for training and set them as CUDA objects.
targets = Variable(torch.FloatTensor(batch_size, 3, 512, 512))
inputs  = Variable(torch.FloatTensor(batch_size, 1, 512, 512))
targets = targets.cuda()
inputs  = inputs.cuda()

##########################

######## Start Training ########

logfile = "Epoch,Training L1,Training L1 Cp,Training L1 X Velocity,Training L1 Y Velocity,Validation L1,Validation L1 Cp,Validation L1 X Velocity,Validation L1 Y Velocity,Time per Epoch\n"

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))
    
    # Start timing block
    t0 = time.time()
    
    #  Initialise L1 parameters.   
    L1_Training_Accumulator = 0.0
    L1_Validation_Accumulator = 0.0

    L1_Training_Cp_Accumulator = 0.0
    L1_Training_X_Vel_Accumulator = 0.0
    L1_Training_Y_Vel_Accumulator = 0.0   

    L1_Validation_Cp_Accumulator = 0.0
    L1_Validation_X_Vel_Accumulator = 0.0
    L1_Validation_Y_Vel_Accumulator = 0.0

    # Inititalise Neural Network object to train
    NeuralNet.train()
    
    # Initialise infinity norm variables.
    Training_Len = len(Training_Loader)
    Training_Batch_Cp_Inf_Norm = np.zeros(Training_Len)
    Training_Batch_Cp_Inf_Norm_X = np.zeros(Training_Len)
    Training_Batch_Cp_Inf_Norm_Y = np.zeros(Training_Len)
    
    Training_Batch_X_Vel_Inf_Norm = np.zeros(Training_Len)
    Training_Batch_X_Vel_Inf_Norm_X = np.zeros(Training_Len)
    Training_Batch_X_Vel_Inf_Norm_Y = np.zeros(Training_Len)
    
    Training_Batch_Y_Vel_Inf_Norm = np.zeros(Training_Len)
    Training_Batch_Y_Vel_Inf_Norm_X = np.zeros(Training_Len)
    Training_Batch_Y_Vel_Inf_Norm_Y = np.zeros(Training_Len)
    
    Validation_Len = len(Validation_Loader)
    Validation_Batch_Cp_Inf_Norm = np.zeros(Validation_Len)
    Validation_Batch_Cp_Inf_Norm_X = np.zeros(Validation_Len)
    Validation_Batch_Cp_Inf_Norm_Y = np.zeros(Validation_Len)
    
    Validation_Batch_X_Vel_Inf_Norm = np.zeros(Validation_Len)
    Validation_Batch_X_Vel_Inf_Norm_X = np.zeros(Validation_Len)
    Validation_Batch_X_Vel_Inf_Norm_Y = np.zeros(Validation_Len)
    
    Validation_Batch_Y_Vel_Inf_Norm = np.zeros(Validation_Len)
    Validation_Batch_Y_Vel_Inf_Norm_X = np.zeros(Validation_Len)
    Validation_Batch_Y_Vel_Inf_Norm_Y = np.zeros(Validation_Len)
    
    # Iterate over every batch in training data.
    for i,Training_Data in enumerate(Training_Loader):
        # Load local batch.
        inputs_cpu, targets_cpu = Training_Data
        # Pad input and target data to form a square image of side length 2^n.
        inputs_cpu = F.pad(inputs_cpu,(0,37,0,208),"constant",0)
        targets_cpu = F.pad(targets_cpu,(0,37,0,208),"constant",0)
        # Remove NaN data and set to 0.
        targets_cpu[targets_cpu != targets_cpu] = 0
        # Set input and target objects as CUDA variables and resize if necessary.
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # Compute Learning Rate decay if active.
        if decayLr:
            currLr = Utility_Functions.computeLR(epoch, epochs, lrNN*0.1, lrNN)
            if currLr < lrNN:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        # Zero the gradients of the network before performing pass.
        NeuralNet.zero_grad()
        # Generate output data from the current network using input batch.
        gen_out = NeuralNet(inputs)

        # Calculate loss from the current pass abd back propagate through network.
        lossL1 = criterionL1(gen_out, targets)
        lossL1[:,0,:,:] = lossL1[:,0,:,:] * inputs[:,0,:,:]
        lossL1[:,1,:,:] = lossL1[:,1,:,:] * inputs[:,0,:,:]
        lossL1[:,2,:,:] = lossL1[:,2,:,:] * inputs[:,0,:,:]
        lossL1 = torch.mean(lossL1[lossL1 != 0])
        
        lossL1_Cp = criterionL1(gen_out[:,0,:,:],targets[:,0,:,:])
        lossL1_Cp = lossL1_Cp * inputs[:,0,:,:]   
        lossL1_Cp = torch.mean(lossL1_Cp[lossL1_Cp != 0])
        
        lossL1_X_Vel = criterionL1(gen_out[:,1,:,:],targets[:,1,:,:])
        lossL1_X_Vel = lossL1_X_Vel * inputs[:,0,:,:]  
        lossL1_X_Vel = torch.mean(lossL1_X_Vel[lossL1_X_Vel != 0])
        
        lossL1_Y_Vel = criterionL1(gen_out[:,2,:,:],targets[:,2,:,:])
        lossL1_Y_Vel = lossL1_Y_Vel * inputs[:,0,:,:]    
        lossL1_Y_Vel = torch.mean(lossL1_Y_Vel[lossL1_Y_Vel != 0])
        
        lossL1.backward()
        
        # Calculate infinity norm and locations.
        Cp_Delta = (gen_out[:,0,:,:] - targets[:,0,:,:]) * inputs[:,0,:,:]
        X_Vel_Delta = (gen_out[:,1,:,:] - targets[:,1,:,:]) * inputs[:,0,:,:]
        Y_Vel_Delta = (gen_out[:,2,:,:] - targets[:,2,:,:]) * inputs[:,0,:,:]
        
        Cp_Infty_Norm = vector_norm(Cp_Delta,float('inf'))
        X_Vel_Infty_Norm = vector_norm(X_Vel_Delta,float('inf'))
        Y_Vel_Infty_Norm = vector_norm(Y_Vel_Delta,float('inf'))
        
        Coord_Cp = torch.where(abs(Cp_Delta) == Cp_Infty_Norm)
        Coord_X_Vel = torch.where(abs(X_Vel_Delta) == X_Vel_Infty_Norm)
        Coord_Y_Vel = torch.where(abs(Y_Vel_Delta) == Y_Vel_Infty_Norm)
        
        if len(Coord_Cp[0]) > 1:
            Cp_X = int(Coord_Cp[1][0])
            Cp_Y = int(Coord_Cp[2][0])
        else:
            Cp_X = int(Coord_Cp[1])
            Cp_Y = int(Coord_Cp[2])  
            
        if len(Coord_X_Vel[0]) > 1:
            X_Vel_X = int(Coord_X_Vel[1][0])
            X_Vel_Y = int(Coord_X_Vel[2][0])
        else:
            X_Vel_X = int(Coord_X_Vel[1])
            X_Vel_Y = int(Coord_X_Vel[2])
            
        if len(Coord_Y_Vel[0]) > 1:            
            Y_Vel_X = int(Coord_Y_Vel[1][0])
            Y_Vel_Y = int(Coord_Y_Vel[2][0])
        else:
            Y_Vel_X = int(Coord_Y_Vel[1])
            Y_Vel_Y = int(Coord_Y_Vel[2])            
        
        Training_Batch_Cp_Inf_Norm[i] = Cp_Infty_Norm
        Training_Batch_Cp_Inf_Norm_X[i] = Cp_X
        Training_Batch_Cp_Inf_Norm_Y[i] = Cp_Y

        Training_Batch_X_Vel_Inf_Norm[i] = X_Vel_Infty_Norm
        Training_Batch_X_Vel_Inf_Norm_X[i] = X_Vel_X
        Training_Batch_X_Vel_Inf_Norm_Y[i] = X_Vel_Y
        
        Training_Batch_Y_Vel_Inf_Norm[i] = Y_Vel_Infty_Norm
        Training_Batch_Y_Vel_Inf_Norm_X[i] = Y_Vel_X
        Training_Batch_Y_Vel_Inf_Norm_Y[i] = Y_Vel_Y

        # If clip is active, clip gradient to limit divergence.
        if Apply_Clip:
            nn.utils.clip_grad_value_(NeuralNet.parameters(), clip)

        # Perform optimization step.
        optimizerG.step()

        # Set loss values to printable items.
        lossL1_Val = lossL1.item()
        lossL1_Cp = lossL1_Cp.item()
        lossL1_X_Vel = lossL1_X_Vel.item()
        lossL1_Y_Vel = lossL1_Y_Vel.item()
        L1_Training_Accumulator += lossL1_Val
        L1_Training_Cp_Accumulator += lossL1_Cp
        L1_Training_X_Vel_Accumulator += lossL1_X_Vel
        L1_Training_Y_Vel_Accumulator += lossL1_Y_Vel
    
        # Print batch number every 10 batches.
        if i % 10 == 0:
            print("Batch {} of {} in epoch {} for training.".format((i + batch_size),len(Training_Loader),(epoch + 1)))

    # Inititalise Neural Network object to evaluate.
    NeuralNet.eval()
    
    # Iterate over every batch in validation data.
    for i,Validation_Data in enumerate(Validation_Loader):
        # Load local batch.
        inputs_cpu, targets_cpu = Validation_Data
        # Pad input and target data to form a square image of side length 2^n.
        inputs_cpu = F.pad(inputs_cpu,(0,37,0,208),"constant",0)
        targets_cpu = F.pad(targets_cpu,(0,37,0,208),"constant",0)
        # Remove NaN data and set to 0.
        targets_cpu[targets_cpu != targets_cpu] = 0
        # Set input and target objects as CUDA variables and resize if necessary.
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # Compute Learning Rate decay if active.
        if decayLr:
            currLr = Utility_Functions.computeLR(epoch, epochs, lrNN*0.1, lrNN)
            if currLr < lrNN:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        # Zero the gradients of the network before performing pass.
        NeuralNet.zero_grad()
        # Generate output data from the current network using input batch.
        gen_out = NeuralNet(inputs)

        # Calculate loss from the current pass abd back propagate through network.
        lossL1 = criterionL1(gen_out, targets)
        lossL1[:,0,:,:] = lossL1[:,0,:,:] * inputs[:,0,:,:]
        lossL1[:,1,:,:] = lossL1[:,1,:,:] * inputs[:,0,:,:]
        lossL1[:,2,:,:] = lossL1[:,2,:,:] * inputs[:,0,:,:]
        lossL1 = torch.mean(lossL1)
        lossL1_Cp = criterionL1(gen_out[:,0,:,:],targets[:,0,:,:])
        lossL1_Cp[0,:,:] = lossL1_Cp[0,:,:] * inputs[0,:,:]
        lossL1_Cp[1,:,:] = lossL1_Cp[1,:,:] * inputs[0,:,:]
        lossL1_Cp[2,:,:] = lossL1_Cp[2,:,:] * inputs[0,:,:]
        lossL1_Cp = torch.mean(lossL1_Cp)        
        lossL1_X_Vel = criterionL1(gen_out[:,1,:,:],targets[:,1,:,:])
        lossL1_X_Vel[0,:,:] = lossL1_X_Vel[0,:,:] * inputs[0,:,:]
        lossL1_X_Vel[1,:,:] = lossL1_X_Vel[1,:,:] * inputs[0,:,:]
        lossL1_X_Vel[2,:,:] = lossL1_X_Vel[2,:,:] * inputs[0,:,:]   
        lossL1_X_Vel = torch.mean(lossL1_X_Vel)
        lossL1_Y_Vel = criterionL1(gen_out[:,2,:,:],targets[:,2,:,:])
        lossL1_Y_Vel[0,:,:] = lossL1_Y_Vel[0,:,:] * inputs[0,:,:]
        lossL1_Y_Vel[1,:,:] = lossL1_Y_Vel[1,:,:] * inputs[0,:,:]
        lossL1_Y_Vel[2,:,:] = lossL1_Y_Vel[2,:,:] * inputs[0,:,:]    
        lossL1_Y_Vel = torch.mean(lossL1_Y_Vel)
        
        # Calculate infinity norm and locations.
        Cp_Delta = (gen_out[:,0,:,:] - targets[:,0,:,:]) * inputs[:,0,:,:]
        X_Vel_Delta = (gen_out[:,1,:,:] - targets[:,1,:,:]) * inputs[:,0,:,:]
        Y_Vel_Delta = (gen_out[:,2,:,:] - targets[:,2,:,:]) * inputs[:,0,:,:]
        
        Cp_Infty_Norm = vector_norm(Cp_Delta,float('inf'))
        X_Vel_Infty_Norm = vector_norm(X_Vel_Delta,float('inf'))
        Y_Vel_Infty_Norm = vector_norm(Y_Vel_Delta,float('inf'))
        
        Coord_Cp = torch.where(abs(Cp_Delta) == Cp_Infty_Norm)
        Coord_X_Vel = torch.where(abs(X_Vel_Delta) == X_Vel_Infty_Norm)
        Coord_Y_Vel = torch.where(abs(Y_Vel_Delta) == Y_Vel_Infty_Norm)

        if len(Coord_Cp[0]) > 1:
            Cp_X = int(Coord_Cp[1][0])
            Cp_Y = int(Coord_Cp[2][0])
        else:
            Cp_X = int(Coord_Cp[1])
            Cp_Y = int(Coord_Cp[2])  
            
        if len(Coord_X_Vel[0]) > 1:
            X_Vel_X = int(Coord_X_Vel[1][0])
            X_Vel_Y = int(Coord_X_Vel[2][0])
        else:
            X_Vel_X = int(Coord_X_Vel[1])
            X_Vel_Y = int(Coord_X_Vel[2])
            
        if len(Coord_Y_Vel[0]) > 1:            
            Y_Vel_X = int(Coord_Y_Vel[1][0])
            Y_Vel_Y = int(Coord_Y_Vel[2][0])
        else:
            Y_Vel_X = int(Coord_Y_Vel[1])
            Y_Vel_Y = int(Coord_Y_Vel[2])   
        
        Validation_Batch_Cp_Inf_Norm[i] = Cp_Infty_Norm
        Validation_Batch_Cp_Inf_Norm_X[i] = Cp_X
        Validation_Batch_Cp_Inf_Norm_Y[i] = Cp_Y

        Validation_Batch_X_Vel_Inf_Norm[i] = X_Vel_Infty_Norm
        Validation_Batch_X_Vel_Inf_Norm_X[i] = X_Vel_X
        Validation_Batch_X_Vel_Inf_Norm_Y[i] = X_Vel_Y
        
        Validation_Batch_Y_Vel_Inf_Norm[i] = Y_Vel_Infty_Norm
        Validation_Batch_Y_Vel_Inf_Norm_X[i] = Y_Vel_X
        Validation_Batch_Y_Vel_Inf_Norm_Y[i] = Y_Vel_Y

        # If clip is active, clip gradient to limit divergence.
        if Apply_Clip:
            nn.utils.clip_grad_value_(NeuralNet.parameters(), clip)

        # Set loss values to printable items.
        lossL1_Val = lossL1.item()
        lossL1_Cp = lossL1_Cp.item()
        lossL1_X_Vel = lossL1_X_Vel.item()
        lossL1_Y_Vel = lossL1_Y_Vel.item()
        L1_Validation_Accumulator += lossL1_Val
        L1_Validation_Cp_Accumulator += lossL1_Cp
        L1_Validation_X_Vel_Accumulator += lossL1_X_Vel
        L1_Validation_Y_Vel_Accumulator += lossL1_Y_Vel
    
        # Print batch number every 10 batches.
        if i % 10 == 0:
            print("Batch {} of {} in epoch {} for validation.".format((i + batch_size),len(Validation_Loader),(epoch + 1)))
    
    
    # Calculate averaged L1 values.
    L1_Training = L1_Training_Accumulator/len(Training_Loader)
    L1_Training_Cp = L1_Training_Cp_Accumulator/len(Training_Loader)
    L1_Training_X_Vel = L1_Training_X_Vel_Accumulator/len(Training_Loader)
    L1_Training_Y_Vel = L1_Training_Y_Vel_Accumulator/len(Training_Loader)
    
    L1_Validation = L1_Validation_Accumulator/len(Validation_Loader)
    L1_Validation_Cp = L1_Validation_Cp_Accumulator/len(Validation_Loader)
    L1_Validation_X_Vel = L1_Validation_X_Vel_Accumulator/len(Validation_Loader)
    L1_Validation_Y_Vel = L1_Validation_Y_Vel_Accumulator/len(Validation_Loader)
    
    # End timing block and calculate time.
    t1 = time.time()
    total_time = t1 - t0
    
    logline = str(epoch + 1) + "," + str(L1_Training) + "," + str(L1_Training_Cp) + "," + str(L1_Training_X_Vel) + "," + str(L1_Training_Y_Vel) + "," + str(L1_Validation) + "," + str(L1_Validation_Cp) + "," + str(L1_Validation_X_Vel) + "," + str(L1_Validation_Y_Vel) + "," + str(total_time) + "\n"
    
    logfile += logline

    # Save network as SaveName.
    print("Writing network to {}.".format(SaveName + "Epoch_{}".format(epoch+1) + ".pt"))
    torch.save(NeuralNet.state_dict(),SaveName + "Epoch_{}".format(epoch+1) + ".pt")
    
    # Save log as .txt file.
    print("Writing log to {}.".format(SaveName+"log.csv"))
    f = open(SaveName+"log.csv", "w")
    f.write(logfile)
    f.close()
    
    # Write infinity norm values and locations to file.
    Training_Infty_Log = "Cp Infinity Norm,Cp X Coordinate,Cp Y Coordinate,X Velocity Infinity Norm,X Velocity X Coordinate,X Velocity Y Coordinate,Y Velocity Infinity Norm,Y Velocity X Coordinate,Y Velocity Y Coordinate\n"
    Validation_Infty_Log = "Cp Infinity Norm,Cp X Coordinate,Cp Y Coordinate,X Velocity Infinity Norm,X Velocity X Coordinate,X Velocity Y Coordinate,Y Velocity Infinity Norm,Y Velocity X Coordinate,Y Velocity Y Coordinate\n"
    
    for k in range(Training_Len):
        Training_Infty_Log += str(Training_Batch_Cp_Inf_Norm[k]) + "," + str(Training_Batch_Cp_Inf_Norm_X[k]) + "," + str(Training_Batch_Cp_Inf_Norm_Y[k]) + "," + str(Training_Batch_X_Vel_Inf_Norm[k]) + "," + str(Training_Batch_X_Vel_Inf_Norm_X[k]) + "," + str(Training_Batch_X_Vel_Inf_Norm_Y[k]) + "," + str(Training_Batch_Y_Vel_Inf_Norm[k]) + "," + str(Training_Batch_Y_Vel_Inf_Norm_X[k]) + "," + str(Training_Batch_Y_Vel_Inf_Norm_Y[k]) + "\n"
        
    for k in range(Validation_Len):
        Validation_Infty_Log += str(Validation_Batch_Cp_Inf_Norm[k]) + "," + str(Validation_Batch_Cp_Inf_Norm_X[k]) + "," + str(Validation_Batch_Cp_Inf_Norm_Y[k]) + "," + str(Validation_Batch_X_Vel_Inf_Norm[k]) + "," + str(Validation_Batch_X_Vel_Inf_Norm_X[k]) + "," + str(Validation_Batch_X_Vel_Inf_Norm_Y[k]) + "," + str(Validation_Batch_Y_Vel_Inf_Norm[k]) + "," + str(Validation_Batch_Y_Vel_Inf_Norm_X[k]) + "," + str(Validation_Batch_Y_Vel_Inf_Norm_Y[k]) + "\n"
    
    j = open(SaveName+"Training_Infinity_Norms/"+"Epoch_{}.csv".format(epoch+1), "w")
    j.write(Training_Infty_Log)
    j.close()

    m = open(SaveName+"Validation_Infinity_Norms/"+"Epoch_{}.csv".format(epoch+1), "w")
    m.write(Validation_Infty_Log)
    m.close()    