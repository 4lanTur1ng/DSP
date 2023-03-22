import numpy as np
import random
import time
import math
from scipy.fftpack import fft,ifft,fftfreq
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl


def dft(se):
    l = len(se)
    t1 = time.time()
    xk = np.array([0 for k in range(0, l)], dtype=complex)
    for j in range(0, l):
        for i in range(0, l):
            tmp = -2 * math.pi * j * i / l
            xk[j] += complex(se[i] * math.cos(tmp), se[i]*math.sin(tmp))
    t2 = time.time()
    print('手动dft运行时间:%s毫秒' % ((t2 - t1) * 1000))
    return xk


def dft1(se):
    t1 = time.time()
    xk = fft(se)
    t2 = time.time()
    print('库函数dft运行时间:%s毫秒' % ((t2 - t1) * 1000))
    return xk


def idft(se):
    l = len(se)
    t1 = time.time()
    xk = np.array([0 for k in range(0, l)], dtype=complex)
    for j in range(0, l):
        for i in range(0, l):
            tmp = 2 * math.pi * j * i / l
            xk[j] += complex(se[i] * math.cos(tmp)/l, se[i] * math.sin(tmp)/l)
    t2 = time.time()
    print('手动idft运行时间:%s毫秒' % ((t2 - t1) * 1000))
    return xk

def idft1(se):
    t1 = time.time()
    xk = ifft(se)
    t2 = time.time()
    print('库函数idft运行时间:%s毫秒' % ((t2 - t1) * 1000))
    return xk

def loss(se1, se2):
    l = len(se1)
    s = 0
    for i in range(l):
        s = (se1[i]-se2[i])**2
    return s/l


se = []
length = 10
start = 0
end = 100
# 生成长度为10，值在0到100之间的随机序列
for i in range(length):
    se.append(random.randint(start, end))
print('原序列为：', se)

# dft
# x1 = dft(se)
# x2 = dft1(se)
# # print(x1)
# # print('#######')
# # print(x2)
# print('DFTloss值',round(float(loss(x1, x2).real), 2))

# idft
# x3 = idft(x1)
# x4 = idft1(x2)
# # print(x3)
# # print('#######')
# # print(x4)
# print('IDFTloss值',round(float(loss(x3, x4).real), 2))



# 验证时域卷积等于频域相乘
h = [-1, 2, -1]
h_start = 0
h_end = 100
h_length = 10
# 生成长度为10，值在0到10之间的序列
# for i in range(h_length):
#     h.append(i)
# print('卷积核为：', h)
# xc = np.convolve(se, h)
#
# n = len(se)+len(h)-1
# N = 2**int((np.log2(n)+1))
# f1 = np.fft.rfft(se, N)
# f2 = np.fft.rfft(h, N)
# f3 = f1*f2
# xc1 = np.fft.irfft(f3)[:n]
#
# print('直接进行卷积后的序列：\n',xc)
# print('#######')
# print('进行DFT，相乘，再进行IDFT后的序列：\n',xc1)
# print('卷积loss值',round(float(loss(xc, xc1).real), 2))

# 采样频率
fs = 4e2

# 生成信号
t = np.arange(0, 160, 1/fs); N = len(t)
# y = 10*np.sin(20*2*np.pi*t)
y = 10*np.sin(170*np.pi*t) + 20*np.sin(370*np.pi*t)+np.random.randn(t.size)

# 傅里叶变换
yf = fft(y)
xf = fftfreq(N, 1/fs)[0:N//2]

# 画图
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel('Frequency (B:Hz)');   plt.ylabel('Amplitude (A)')
plt.grid()
plt.show()


