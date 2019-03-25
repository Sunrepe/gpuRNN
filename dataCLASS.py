import os
import numpy as np

class Data(object):
	"""dshapetring for Data"""
	def __init__(self, shape):
		super(Data, self).__init__()
		self.argshape = shape

	def getnum(self):
		return np.zeros(self.argshape)

def main():
	# print(os.getcwd(),os.listdir()) 
	print("12345")

if __name__ == '__main__':
	main()