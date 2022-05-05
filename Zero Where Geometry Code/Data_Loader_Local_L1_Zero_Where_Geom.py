import numpy as np
from torch.utils.data import Dataset

# Load individual datapoints for lazy loading with CLF.
def IndividualTrainingSampleLoader_CLF(sample_file,ID,CLF_Name,BC_Name,L1Def,normalisation=True,Cp_Norm=25,X_Vel_Norm=130,Y_Vel_Norm=130):
    
    # Initialise sample objects.
    Training_Sample_Input = np.zeros([1,512,512])
    Training_Sample_Target = np.zeros([3,512,512])
    Training_CLF = np.zeros([1,512,512])
    LocalL1 = np.zeros([512,512])
    
    # Load sample files and add to batch sample objects.
    Geom_File = sample_file + "/Geom/Geom_{}.npy".format(ID) 
    Cp_File = sample_file + "/Cp/Cp_{}.npy".format(ID) 
    X_Vel_File = sample_file + "/X_Vel/X_Vel_{}.npy".format(ID) 
    Y_Vel_File = sample_file + "/Y_Vel/Y_Vel_{}.npy".format(ID) 
    CLF_File = sample_file + "/{}/CLF_{}.npy".format(CLF_Name,ID) 
    BC_File = sample_file + "/{}/Cp_BC_{}.npy".format(BC_Name,ID) 
    LocalL1_File = sample_file + L1Def + "Local_L1_{}.npy".format(ID) 
    
    Geom_Data = np.load(Geom_File)
    Cp_Data = np.load(Cp_File)
    X_Vel_Data = np.load(X_Vel_File)
    Y_Vel_Data = np.load(Y_Vel_File)
    CLF_Data = np.load(CLF_File)
    BC_Data = np.load(BC_File)
    LocalL1_Data = np.load(LocalL1_File)
    
    # Fix errors in geometry data.
    Geom_Data[300:,:] = True
    
    # Find where geometry is located in geometry array.
    zero_locations = np.where(Geom_Data == 0)
    
    # Find padding lengths.
    left = np.min(zero_locations[1])
    up = np.min(zero_locations[0])
    right = np.max(zero_locations[1])
    down = np.max(zero_locations[0])
    pad_left = left
    pad_up = up
    pad_right = len(Geom_Data[0,:]) - right - 1
    pad_down = len(Geom_Data[:,0]) - down - 1
    
    # Pad arrays with required values.
    BC_Data = np.pad(BC_Data,((pad_up,pad_down),(pad_left,pad_right)))
    CLF_Data = np.pad(CLF_Data,((pad_up,pad_down),(pad_left,pad_right)))
    
#    Training_CLF[0,:len(CLF_Data[:,0]),:len(CLF_Data[0,:])] = CLF_Data + Geom_Data
    
    Training_Sample_Input[0,:len(Geom_Data[:,0]),:len(Geom_Data[0,:])] = Geom_Data
    
    Cp_Data = np.nan_to_num(Cp_Data)
    X_Vel_Data = np.nan_to_num(X_Vel_Data)
    Y_Vel_Data = np.nan_to_num(Y_Vel_Data)
    
#    Cp_Data += BC_Data
    
    # Apply normalisation if required.
    if normalisation:
        Cp_Data /= Cp_Norm
        X_Vel_Data /= X_Vel_Norm
        Y_Vel_Data /= Y_Vel_Norm       
    
    Training_Sample_Target[0,:len(Cp_Data[:,0]),:len(Cp_Data[0,:])] = Cp_Data
    Training_Sample_Target[1,:len(X_Vel_Data[:,0]),:len(X_Vel_Data[0,:])] = X_Vel_Data
    Training_Sample_Target[2,:len(Y_Vel_Data[:,0]),:len(Y_Vel_Data[0,:])] = Y_Vel_Data
    
    LocalL1[:len(LocalL1_Data[:,0]),:len(LocalL1_Data[0,:])] = LocalL1_Data
    
    Unique_CLF_Values = np.unique(CLF_Data)
    
    Unique_CLF_Values[::-1].sort()
    
    Unique_CLF_Values = Unique_CLF_Values[:-1]
    
    for i in range(len(Geom_Data[0,:])):
        
        Cp_BC = Cp_Data[-1,i]
        CLF_BC = Unique_CLF_Values.T
        
        Training_Sample_Target[0,len(Geom_Data[:,0]):len(Geom_Data[:,0])+len(CLF_BC),i] = Cp_BC
        Training_Sample_Target[1,len(Geom_Data[:,0]):len(Geom_Data[:,0])+len(CLF_BC),i] = 29.162/X_Vel_Norm
        Training_Sample_Target[2,len(Geom_Data[:,0]):len(Geom_Data[:,0])+len(CLF_BC),i] = 0
        
#        Training_CLF[0,len(Geom_Data[:,0]):len(Geom_Data[:,0])+len(CLF_BC),i] = CLF_BC
        
    for i in range(len(Geom_Data[:,0])):
        
        Cp_BC = Cp_Data[i,-1]
        X_Vel_BC = X_Vel_Data[i,-1]
        Y_Vel_BC = Y_Vel_Data[i,-1]
        CLF_BC = Unique_CLF_Values
        
        Training_Sample_Target[0,i,len(Geom_Data[0,:]):len(Geom_Data[0,:])+len(CLF_BC)] = Cp_BC
        Training_Sample_Target[1,i,len(Geom_Data[0,:]):len(Geom_Data[0,:])+len(CLF_BC)] = X_Vel_BC
        Training_Sample_Target[2,i,len(Geom_Data[0,:]):len(Geom_Data[0,:])+len(CLF_BC)] = Y_Vel_BC
        
#        Training_CLF[0,i,len(Geom_Data[0,:]):len(Geom_Data[0,:])+len(CLF_BC)] = CLF_BC
    
    return Training_Sample_Input, Training_Sample_Target, Training_CLF, LocalL1


# Initialise MultiFoilData dataset class.
class MultiFoilData_CLF(Dataset):
    
    def __init__(self,root_dir,Sample_Dictionary,ID_Dictionary,CLF_Name,BC_Name,L1Def):
        
        # Initialise the root directory and dataset index dictionary.
        print("The dataset root directory is {}.".format(root_dir))
        self.root_dir = root_dir
        self.Samples = Sample_Dictionary
        self.IDs = ID_Dictionary
        self.CLF_Name = CLF_Name
        self.BC_Name = BC_Name
        self.L1Def = L1Def
    
    def __getitem__(self,index):
        
        # Load in the input and target data from the specified file using IndividualTrainingSampleLoader.
        file = self.root_dir + self.Samples[index]
        inputs, targets, CLF, LocalL1 = IndividualTrainingSampleLoader_CLF(file,self.IDs[index],self.CLF_Name,self.BC_Name,self.L1Def)
        return inputs, targets, CLF, LocalL1
    
    def __len__(self):
        
        # Find length of dataset.
        length = len(self.IDs)
        return length