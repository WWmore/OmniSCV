"""
https://stackoverflow.com/questions/76080798/is-it-possible-to-map-an-image-texture-onto-a-mesh-using-pyqtgraph
"""
from OpenGL.GL import *  # noqa
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from scipy.ndimage import gaussian_filter


def to_xyz(phi, theta):
    theta = np.pi - theta
    r = 10.0
    xpos = r * np.sin(theta) * np.cos(phi)
    ypos = r * np.sin(theta) * np.sin(phi)
    zpos = r * np.cos(theta)
    return xpos, ypos, zpos


class GLTexturedSphereItem(GLGraphicsItem):
    def __init__(self, data, smooth=False, glOptions="translucent", parentItem=None):
        """

        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 3D numpy array (x, y, RGBA) with dtype=ubyte.
                        (See functions.makeRGBA)
        smooth          (bool) If True, the volume slices are rendered with linear interpolation
        ==============  =======================================================================================
        """

        self.smooth = smooth
        self._needUpdate = False
        super().__init__(parentItem=parentItem)
        self.setData(data)
        self.setGLOptions(glOptions)
        self.texture = None

    def initializeGL(self):
        if self.texture is not None:
            return
        glEnable(GL_TEXTURE_2D)
        self.texture = glGenTextures(1)

    def setData(self, data):
        self.data = data
        self._needUpdate = True
        self.update()

    def _updateTexture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        if self.smooth:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        # glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        shape = self.data.shape

        ## Test texture dimensions first
        glTexImage2D(
            GL_PROXY_TEXTURE_2D,
            0,
            GL_RGBA,
            shape[0],
            shape[1],
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            None,
        )
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception(
                "OpenGL failed to create 2D texture (%dx%d); too large for this hardware."
                % shape[:2]
            )

        data = np.ascontiguousarray(self.data.transpose((1, 0, 2)))
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            shape[0],
            shape[1],
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data,
        )
        glDisable(GL_TEXTURE_2D)

    def paint(self):
        if self._needUpdate:
            self._updateTexture()
            self._needUpdate = False
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        self.setupGLState()

        glColor4f(1, 1, 1, 1)

        theta = np.linspace(0, np.pi, 32, dtype="float32")
        phi = np.linspace(0, 2 * np.pi, 64, dtype="float32")
        t_n = theta / np.pi
        p_n = phi / (2 * np.pi)

        glBegin(GL_QUADS)
        for j in range(len(theta) - 1):
            for i in range(len(phi) - 1):
                xyz_nw = to_xyz(phi[i], theta[j])
                xyz_sw = to_xyz(phi[i], theta[j + 1])
                xyz_se = to_xyz(phi[i + 1], theta[j + 1])
                xyz_ne = to_xyz(phi[i + 1], theta[j])

                glTexCoord2f(p_n[i], t_n[j])
                glVertex3f(xyz_nw[0], xyz_nw[1], xyz_nw[2])
                glTexCoord2f(p_n[i], t_n[j + 1])
                glVertex3f(xyz_sw[0], xyz_sw[1], xyz_sw[2])
                glTexCoord2f(p_n[i + 1], t_n[j + 1])
                glVertex3f(xyz_se[0], xyz_se[1], xyz_se[2])
                glTexCoord2f(p_n[i + 1], t_n[j])
                glVertex3f(xyz_ne[0], xyz_ne[1], xyz_ne[2])

        glEnd()
        glDisable(GL_TEXTURE_2D)


app = pg.mkQApp("GLImageItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle("pyqtgraph example: GLImageItem")
w.setCameraPosition(distance=200)

a = np.zeros((256, 256))
N = 100
x = (np.random.random(N) * 256).astype(int)
y = (np.random.random(N) * 256).astype(int)

a[y, x] = 10000

smooth = gaussian_filter(a, sigma=5, mode="wrap").reshape(256, 256, 1)
smooth = (smooth / smooth.max() * 255).astype(np.uint8)
smooth = np.broadcast_to(smooth, (256, 256, 4)).copy()
smooth[..., 3] = 255
smooth[..., 0] += 50

v1 = GLTexturedSphereItem(np.clip(smooth, 0, 255))
w.addItem(v1)

if __name__ == "__main__":
    pg.exec()