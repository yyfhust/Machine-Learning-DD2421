import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class svm:

	def __init__ (self, kernel_type="linear",p=2,sigma=1,C=None):
	## type: kernel type , default is linear
		# p: parameter for Polynomial kernels , default is 2;p = 2 will make quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher will result in more complex shapes.
		# sigma: parameter for rbf, used to control the smoothness of the boundary.
		self.kernel_type=kernel_type
		self.p=p
		self.sigma=sigma
		self.C=C

	def kernel(self, x1, x2):
		##The kernel function takes two data points as arguments and returns a “scalar product-like” similarity measure; a scalar value. 
		
		if self.kernel_type== "linear":
			return np.dot(x1,x2)

		elif self.kernel_type =="poly":
			return np.power(np.dot(x1,x2)+1,self.p)

		elif self.kernel_type =="rbf":
			return np.exp(-((np.linalg.norm(x1-x2))**2)/(2*self.sigma**2))
			

	def objective(self, alpha):

		## objective is a function you have to define, 
		## which takes a vector α⃗ as argu- ment and returns a scalar value, 

		alpha1= alpha[:,np.newaxis]
		alpha2=np.transpose(alpha1)

		return 0.5 * np.sum(self.P_matrix * np.dot(alpha1,alpha2))  -  np.sum(alpha)

		

	def Pij(self):
		## pre-compute a matrix : ti*tj*K(xi,xj)
		self.P_matrix=np.zeros([self.x.shape[0],self.x.shape[0]])

		for i in range(self.x.shape[0]):
			for j in range(self.x.shape[0]):
				self.P_matrix[i][j] = self.t[i]*self.t[j]* self.kernel(self.x[i],self.x[j])

	def zerofun(self, alpha):
		return np.sum( alpha*self.t )

	def fit(self,x,t):
		self.x=x
		self.t=t
		self.Pij()

		## minimize
		ret = minimize( self.objective , np.zeros(self.x.shape[0]) , bounds=[(0, self.C) for b in range(self.x.shape[0])], constraints={"type":'eq','fun':self.zerofun} )
		if (ret["success"] == True):
			self.alpha = ret['x']
		else:
			print ("minimize failed")

		## compute b 
		for index,sv in enumerate(self.alpha):
			if sv>10e-5:  ## !=0
				break

		self.b=0
		for i in range(self.x.shape[0]):
			self.b = self.b+ self.t[i] * self.alpha[i] *self.kernel(self.x[index],self.x[i])
		self.b=self.b-t[index]


	def predict(self, s):
		prediction=0
		for i in range(self.x.shape[0]):
			prediction=prediction+ self.alpha[i] * self.t[i] * self.kernel(s,self.x[i])

		prediction=prediction-self.b
		return prediction









