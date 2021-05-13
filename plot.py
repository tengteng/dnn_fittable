from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection="3d")

N = 5

# Make H surface
X = np.linspace(-1.25, 1.25, 1024)
Y = np.linspace(-1.25, 1.25, 1024)
X, Y = np.meshgrid(X, Y)
Z = 1 / N * np.abs(1 - (X + Y * 1j) ** -N) / np.abs(1 - 1 / (X + Y * 1j))
Z = 20 * np.log10(Z)

# Make the image of the unit circle
omega = np.linspace(0, 2 * np.pi, 2048)
circ_X = np.cos(omega)
circ_Y = np.sin(omega)
circ_Z = 1 / N * np.sin(N * omega / 2) / np.sin(omega / 2)
circ_Z = 10 * np.log10(circ_Z ** 2) + 1

# Plot the H surface and the unit circle
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=True)
circ = ax.plot(circ_X, circ_Y, circ_Z, color="green")

ax.set_zlim(-40, 10)
plt.show()
