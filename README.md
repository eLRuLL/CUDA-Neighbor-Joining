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
  
  /************** DATA-PARALLELIZABLE using CUDA **************/
  1. Calculate the tree-divergence for every object (an array). 
      r[i] = M[i][0] + M[i][1] + M[i][2] + ... + M[i][N-1];
          where i = 0,2,3.... N-1.
  /************************************************************/
  
  /************** DATA-PARALLELIZABLE using CUDA **************/
  2. Calculate a new matrix (Mt) of distances (temporal) with the equation:
      Mt[i][j] = M[i][j] - (r[i] + r[j])/(N - 2);
  /************************************************************/
  
  3. Select the objects "i" and "j" where Mt[i][j] is the minimum.
  4. Create an new object U that replace "i" and "j"
  5. Calculate the distances between i -> U and j -> U with:
      S_iU = M[i][j]/2 + (r[i] - r[j])/(2*(N - 2));
      S_jU = M[i][j] - S_iU;

  /************** DATA-PARALLELIZABLE using CUDA **************/
  6. Calculate the distances between the new object U and the rest:
      M[k][U] = (M[i][k] + M[j][k] - M[i][j])/2;
          where k!=i ^ k!=j
  /************************************************************/

  7. N = N - 1;