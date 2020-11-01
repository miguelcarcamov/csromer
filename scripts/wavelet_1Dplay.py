import pywt
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

x = signal.gaussian(64, std=4)
print("Signal:", x)
x += 0.01*np.random.normal(size=64, loc=0, scale=1.0)
w = pywt.Wavelet('db2')
cA, cD = pywt.dwt(x, wavelet=w, mode='periodization')
print("Coeffs:" , cA, cD)
#print(pywt.waverec(coeffs, w, mode='periodization'))
plt.figure(1)
plt.title("Signal")
plt.plot(x)
plt.figure(2)
plt.title("Coeffs Approx")
plt.plot(cA)
plt.figure(3)
plt.title("Coeffs Detail")
plt.plot(cD)
cA = pywt.threshold(cA, 1e-2, mode='soft')
cD = pywt.threshold(cD, 1e-2, mode='soft')
x_back = pywt.idwt(cA, cD, wavelet=w, mode='periodization')
plt.figure(4)
plt.title("Signal back")
plt.plot(x_back)
plt.show()
#print("Signal back: ", x_back)
