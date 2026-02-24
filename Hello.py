print("welcome to coke world")
import numpy as np 
np.random.seed(0)
import matplotlib.pyplot as plt
x = np.random.rand(100)
y = np.random.rand(100)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Random Points')
plt.show()
