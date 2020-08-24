A lightweight, no dependencies C++ Implementation of SMPL - A Skinned Multi-Person Linear Model.

## Overview

Project based on https://github.com/YeeCY/SMPLpp repository.

This project implements a 3D human skinning model - SMPL: A Skinned
Multi-Person Linear Model with C++. The official SMPL model is available at http://smpl.is.tue.mpg.de.

It allows to set shape and pose for SMPL models like whowle body model, hand model (MANO hand), head model (FLAME model).
Such models are usually named morphable models and used in 3d fitting applications, like head or hand pose and shape estimation.

On my machine mesh generation takes:

- MANO_hand - 0.5 ms
- FLAME head - 5 ms
- female model - 10 ms

## Prerequisites

- OS
  Windows 10
  MSVS 2019 

- Packages   
2. [CMake](https://cmake.org/download/): A tool to build, test and pack up 
   C++ program.

## Model preprocessing ##

You need to preprocess initial pkl model format to npz using script from subfolder scripts/preprocess.py and copy result npz file to exe's folder to model subfolder.

![body](ManoHand.gif)
![hand](FLAME.gif)
![head](female.gif)