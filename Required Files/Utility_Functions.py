import math


# Denormalize data.
def DeNormaliser(data,Cp_Norm=25,X_Vel_Norm=130,Y_Vel_Norm=130):
    
    data[:,0,:,:] *= Cp_Norm
    data[:,1,:,:] *= X_Vel_Norm
    data[:,2,:,:] *= Y_Vel_Norm
    
    DeNormalised_Data = data
    
    return DeNormalised_Data


# Computes learning rate based on iteration and epochs.
def computeLR(i,epochs, minLR, maxLR):
    
    if i < epochs*0.5:
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR-minLR)*f
