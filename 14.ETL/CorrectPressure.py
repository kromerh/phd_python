from sklearn.linear_model import LinearRegression
import numpy as np

class CorrectPressure():

	def __init__(self):
		self.p_display = np.asarray([9.5e-6, 9.6e-6, 9.8e-6, 1.0e-5, 5.6e-6, 5.3e-6, 5.0e-6, 4.7e-6, 3.9e-6], float)
		self.p_py = np.asarray([1.01e-5, 1.05e-5, 1.09e-5, 1.13e-5, 5.91e-6, 5.68e-6, 5.26e-6, 5.06e-6, 4.18e-6], float)

	def int_pol_pressure(self, p):
		"""
		takes pressure values as input (numpy float array). These should be the values stored in "pressure_IS_corrected" in the pressure storage table.
		returns the correct pressure values
		"""

		# convert into linear space
		X = np.log(self.p_py).reshape(-1,1)
		y = np.log(self.p_display).reshape(-1,1)

		reg = LinearRegression().fit(X, y)
		m = reg.coef_[0][0]
		c = reg.intercept_[0]

		# conver to log
		p = np.log(p)
		# use correction
		p_out = reg.predict(p.reshape(-1,1))
		# transform back
		p_out = np.exp(p_out)

		return p_out