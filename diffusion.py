from scipy.fftpack import dct, idct
import numpy as np


#  forward process
def heat_eq_forward(u, w, h, t):
    u_proj = dct(u, axis=0, norm='ortho')
    u_proj = dct(u_proj, axis=1, norm='ortho')
    freq_w = np.linspace(0, w - 1, w)
    freq_h = np.linspace(0, h - 1, h)
    freq = -np.pi ** 2 * ((freq_w[:, None] / w) ** 2 + (freq_h[None, :] / h) ** 2)
    u_proj = np.exp(freq * t) * u_proj
    ut = idct(u_proj, axis=0, norm='ortho')
    ut = idct(ut, axis=1, norm='ortho')
    return ut

# u = [[1, 1], [2, 2]]
# print(heat_eq_forward(u, 2, 2, 0))
