from column_generation import column_generation
import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
labels = [1,-1,-1]

column_generation(x, labels, 1)
