import numpy as np
from torch.utils.data import Dataset

# Example File Location: "D:/IP/Training_Data/Sample_1_1"


# Old batch sample loader.
def TrainingSampleLoader(sample_file,normalisation=True,Cp_Norm=25,X_Vel_Norm=130,Y_Vel_Norm=130):
    
    
    # Initialise sample objects.
    Training_Sample_Inputs = np.zeros([1000,1,304,475])
    Training_Sample_Targets = np.zeros([1000,3,304,475])
    
    # Calculate offset value.
    offset_value = ((int(sample_file[-3]) - 1) * 5000) + ((int(sample_file[-1]) - 1) * 1000)
    
    # Load sample files and add to batch sample objects.
    for i in range(1,1001):
        Geom_File = sample_file + "/Geom/Geom_{}.npy".format(i + offset_value) 
        Cp_File = sample_file + "/Cp/Cp_{}.npy".format(i + offset_value) 
        X_Vel_File = sample_file + "/X_Vel/X_Vel_{}.npy".format(i + offset_value) 
        Y_Vel_File = sample_file + "/Y_Vel/Y_Vel_{}.npy".format(i + offset_value) 
        
        Geom_Data = np.load(Geom_File)
        Cp_Data = np.load(Cp_File)
        X_Vel_Data = np.load(X_Vel_File)
        Y_Vel_Data = np.load(Y_Vel_File)
        
        Training_Sample_Inputs[i-1,0,:,:] = Geom_Data
        
        # Apply normalisation if required.
        if normalisation:
            Cp_Data /= Cp_Norm
            X_Vel_Data /= X_Vel_Norm
            Y_Vel_Data /= Y_Vel_Norm       
        
        Training_Sample_Targets[i-1,0,:,:] = Cp_Data
        Training_Sample_Targets[i-1,1,:,:] = X_Vel_Data
        Training_Sample_Targets[i-1,2,:,:] = Y_Vel_Data
        
    return Training_Sample_Inputs, Training_Sample_Targets


# Load individual datapoints for lazy loading.
def IndividualTrainingSampleLoader(sample_file,ID,normalisation=True,Cp_Norm=25,X_Vel_Norm=130,Y_Vel_Norm=130):
    
    # Initialise sample objects.
    Training_Sample_Input = np.zeros([1,304,475])
    Training_Sample_Target = np.zeros([3,304,475])
    
    # Load sample files and add to batch sample objects.
    Geom_File = sample_file + "/Geom/Geom_{}.npy".format(ID) 
    Cp_File = sample_file + "/Cp/Cp_{}.npy".format(ID) 
    X_Vel_File = sample_file + "/X_Vel/X_Vel_{}.npy".format(ID) 
    Y_Vel_File = sample_file + "/Y_Vel/Y_Vel_{}.npy".format(ID) 
    
    Geom_Data = np.load(Geom_File)
    Cp_Data = np.load(Cp_File)
    X_Vel_Data = np.load(X_Vel_File)
    Y_Vel_Data = np.load(Y_Vel_File)
    
    # Fix errors in geometry data.
    Geom_Data[300:,:] = True
    
    Training_Sample_Input[0,:,:] = Geom_Data
    
    # Apply normalisation if required.
    if normalisation:
        Cp_Data /= Cp_Norm
        X_Vel_Data /= X_Vel_Norm
        Y_Vel_Data /= Y_Vel_Norm       
    
    Training_Sample_Target[0,:,:] = Cp_Data
    Training_Sample_Target[1,:,:] = X_Vel_Data
    Training_Sample_Target[2,:,:] = Y_Vel_Data
        
    return Training_Sample_Input, Training_Sample_Target

# Initialise MultiFoilData dataset class.
class MultiFoilData(Dataset):
    
    def __init__(self,root_dir,Sample_Dictionary,ID_Dictionary):
        
        # Initialise the root directory and dataset index dictionary.
        print("The dataset root directory is {}.".format(root_dir))
        self.root_dir = root_dir
        self.Samples = Sample_Dictionary
        self.IDs = ID_Dictionary
    
    def __getitem__(self,index):
        
        # Load in the input and target data from the specified file using IndividualTrainingSampleLoader.
        file = self.root_dir + self.Samples[index]
        inputs, targets = IndividualTrainingSampleLoader(file,self.IDs[index])
        return inputs, targets
    
    def __len__(self):
        
        # Find length of dataset.
        length = len(self.IDs)
        return length