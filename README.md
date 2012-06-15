CUDA-Neighbor-Joining
=====================

A easy to understand Neighbor Joining algorithm made with CUDA


Neighbor - Joining Algorithm
____________________________

Initialization:

N = Number of objets.
Create a matrix M with NxN elements (we'll only use a triangular matrix because
  this matrix represents the distances between the objets).

While N > 2 Then
  
  1. Calculate the tree-divergence for every object.
      r_i = M[i][0] + M[i][1] + M[i][2] + ... + M[i][N-1];
  
  2. Calculate a new matrix of distances (temporal) with the equation:
                    