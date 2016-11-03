from sklearn.svm import SVR
import numpy as np
import math

def rbf(gamma=1.0):
	"""
	Argments:	gamma (defaut 1.0)
	Returns:	a callable which returns math.exp((np.linalg.norm(x1-x2))*(-1.0*gamma)) for some x1,x2
	"""
	def rbf_fun(x1,x2):
		return math.exp((np.linalg.norm(x1-x2))*(-1.0*gamma))
	return rbf_fun

def lin(offset=0):
	"""
	Argments:	offset (defaut 0)
	Returns:	a callable which returns x1.dot(x2.transpose())+offset for some x1,x2
	"""
	def lin_fun(x1,x2):
		return x1.dot(x2.transpose())+offset
	return lin_fun

def poly(power=3,offset=0):
	"""
	Argments:	power (default 3) 
			offset (defaut 0)
	Returns:	a callable which returns math.pow(x1.dot(x2.transpose())+offset,power) for some x1,x2
	"""
	def poly_fun(x1,x2):
		return math.pow(x1.dot(x2.transpose())+offset,power)
	return poly_fun

def sig(alpha=1.0,offset=0):
	"""
	Argments:	alpha (default 1.0) 
			offset (defaut 0)
	Returns:	a callable which returns math.tanh(alpha*1.0*x1.dot(x2.transpose())+offset) for some x1,x2
	"""
	def sig_fun(x1,x2):
		return math.tanh(alpha*1.0*x1.dot(x2.transpose())+offset)
	return sig_fun

def kernel_matrix(x,kernel):
	mat=np.zeros((x.shape[0],x.shape[0]))
	for a in range(x.shape[0]):
		for b in range(x.shape[0]):
			mat[a][b]=kernel(x[a],x[b])
	return mat

def f_dot(kernel_mat1,kernel_mat2):
	return (kernel_mat1.dot(kernel_mat2.transpose())).trace()

def A(kernel_mat1,kernel_mat2):
	return (f_dot(kernel_mat1,kernel_mat2))/(math.sqrt(f_dot(kernel_mat1,kernel_mat1)*f_dot(kernel_mat2,kernel_mat2)))

def beta_finder(x,y,kernel_list):
	"""
	Argments:	x (training data inputs) (as a numpy matrix)
			y (training data outputs) (as a numpy array/matrix of shape(n,1)) Supports only singly output as for now
			kernel lists (a list of kernal functions)
	Returns:	A list in the order of kernels indicating weight of every kernel
	"""
	y=np.matrix(y).reshape(y.shape[0],1)
	yyT=y.dot(y.transpose())
	deno=sum([A(kernel_matrix(x,kernel),yyT) for kernel in kernel_list])
	betas=[A(kernel_matrix(x,kernel),yyT)/deno for kernel in kernel_list]
	return [float(b) for b in betas]

def multi_kernel_maker(x,y,kernel_list):
	"""
	Argments:	x (training data inputs) (as a numpy matrix)
			y (training data outputs) (as a numpy array/matrix of shape(n,1)) Supports only singly output as for now
			kernel lists (a list of kernal functions)
	Returns:	a callable, which returns the kernal matrixes of the input

	Example Use Case:
		x=np.matrix([[1,2],[2,4],[3,6],[4,8],[5,10]])
		y=np.array([2,4,6,8,10])
		a=multi_kernel_maker(x,y,[lin(),rbf(),sig(),poly(power=2)])
		svr=SVR(kernel=a)
		c=svr.fit(x,y)
	"""
	betas=beta_finder(x,y,kernel_list)
	print(betas)
	def multi_kernal(x1,x2):
		"""
		Arguments: 	x1,x2 inputs for kernel function

		Returns:	Kernel Matrix of x1 & x2
		"""
		mat=np.zeros((x1.shape[0],x2.shape[0]))
		for a in range(x1.shape[0]):
			for b in range(x2.shape[0]):
				mat[a][b]=sum([betas[i]*kernel(x1[a],x2[b]) for i,kernel in enumerate(kernel_list)])
		return mat
	return multi_kernal
