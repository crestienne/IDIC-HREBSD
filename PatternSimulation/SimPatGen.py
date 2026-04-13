'''
This file enables the generation of EBSD patterns using EBSDtorch. It defines a class patternSimulation that initializes the necessary parameters and functions to generate EBSD patterns based on input Euler angles and pattern center. The class includes methods for setting up the master pattern, converting Euler angles and pattern center to SE3 vectors, and generating the EBSD pattern using the project_hrebsd function from EBSDtorch. The generated pattern can then be used for further analysis or optimization. 
'''

#importing the necessary libraries 

import os 
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch import Tensor
import torch.nn as nn

import h5py

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ebsdtorch"))

from ebsdtorch.ebsd import geometry
from ebsdtorch.io.read_master_pattern import read_master_pattern
from ebsdtorch.ebsd.geometry import bruker_geometry_to_SE3
from ebsdtorch.ebsd.project_hrebsd import project_hrebsd, hrebsd_coords
from ebsdtorch.lie_algebra.se3 import se3_exp_map_split, se3_log_map_split
from ebsdtorch.s2_and_so3.orientations import bu2qu, qu2bu
from ebsdtorch.s2_and_so3.quaternions import qu_apply
from typing import Optional, Tuple


#=====================Class for Orientation Optimization =====================

class patternSimulation():
    def __init__(self):

        self.batch = 1 #defining the batch size - currently set to 1, but can be changed to allow GPU parallelization
        self.device = torch.device("cpu")
        self.dtype = torch.float32  # Specify the data type

        #Defining the size/shape of detector
        self.detector_height = 516
        self.detector_width = 516
        self.det_shape = (self.detector_height, self.detector_width)
        self.detector_tilt_deg = 70

        self.azimuthal_deg = 0.0
        self.sample_tilt_deg =  0.0 

        self.Dinv = torch.eye(3, dtype=torch.float32).to(self.device)[None, :, :] #no deformation is considered, is D is just identity matrix

        self.se3_mask = torch.ones(6, device=self.device, dtype=self.dtype)
        self.se3_vector = torch.zeros(100, 6, device= self.device, dtype=self.dtype) #this is what I want to alter to optimize the orientation

        self.quats = torch.zeros(4, device= self.device, dtype=self.dtype) #this is the initial orientation guess

        
    def mastersetup(self, masterpatternpath):

        self.master_pattern = read_master_pattern(masterpatternpath).to(self.device).to(self.dtype)
        print("Master Pattern Loaded")

    def EandPCSet(self, Euler, PC, D=None, verbose=True):
        '''
        Argument:
        -Takes in a single Euler angle array in Edax TSL format(numpy array), E is in radians - this is the initial orientation guess and PC (3 values)
        -converts E to quaternions
        -sets the class values so that gen pattern can be called

        Returns: None
        '''

        #convert E to a tensor
        E = torch.tensor(Euler, dtype=torch.float32, device =self.device)[None, :]
        self.quats = bu2qu(E) #convert the Euler angles to quaternions
        self.quats = self.quats.repeat(self.batch, 1) #repeat the quaternions for the batch size

        #define self.quats to be a parameter
        self.quats = torch.nn.Parameter(self.quats) #this allows the quaternions to be optimized



        #convert PC to torch tensor
        self.pattern_centerInit = torch.tensor(PC, dtype=torch.float32, device =self.device)[None, :]

        #convert the pattern center and tilt angles to SE3 vector
        rotation, translation = bruker_geometry_to_SE3(
            pattern_centers= self.pattern_centerInit,
            primary_tilt_deg = torch.tensor([-(self.detector_tilt_deg - self.sample_tilt_deg)], device=self.device, dtype=self.dtype),
            secondary_tilt_deg=torch.tensor([self.azimuthal_deg], device=self.device, dtype=self.dtype),
            detector_shape= self.det_shape,
        )

        self.se3_vector = se3_log_map_split(rotation, translation) #combine the rotation and translation into a single vector
        self.se3_vector = self.se3_vector.repeat(self.batch, 1)  #repeat the SE3 vector for the batch size

        #Make SE3 vector a parameter
        self.se3_vector = torch.nn.Parameter(self.se3_vector) #this allows the SE3 vector to be optimized

        if verbose:
            print('Initial Quaternions: ' + str(self.quats))
            print('Initial SE3 Vector: ' + str(self.se3_vector))



    def GenPattern(self):

        #normalize the quaternions - only want to optimize unit quaternions so this forces the optimizer only to optimize the rotation 
        quats = self.quats / torch.norm(self.quats, dim=1, keepdim=True)



        f_inverse = self.Dinv

            
        coords = hrebsd_coords(
            self.se3_mask, self.se3_vector, quats, f_inverse, self.det_shape, 1, self.device, self.dtype
        )

        pats = project_hrebsd(self.master_pattern, coords)

        return pats


