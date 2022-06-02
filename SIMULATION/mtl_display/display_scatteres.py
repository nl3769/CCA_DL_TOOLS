import os
import glob
import matplotlib.pyplot as plt
from mat4py import loadmat
import nibabel as nib
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

class displayScatteres():

    def __init__(self, pres):

        self.pres = pres
        self.directories = self.find_dir()
        # self.phantom = self.load_phantom()
        self.OF = self.load_OF()
        self.image_info = self.get_image_info()

    # ------------------------------------------------------------------------------------------------------------------
    def find_dir(self):
        dnames = sorted(os.listdir(self.pres))

        path_data = {}
        # --- we need flow, phantom and image
        substr = "phantom"

        for name in dnames:
            path_data[name] = {}

            # -- path to flow
            flow_name = sorted(glob.glob(os.path.join(self.pres, name, substr, "OF*.nii")))
            if flow_name:
                path_data[name]["OF"] = os.path.join(self.pres, name, substr, flow_name[0])
            else:
                path_data[name]["OF"] = []
            # -- path to image
            I_name = sorted(glob.glob(os.path.join(self.pres, name, substr, "image_information*.mat")))
            if I_name:
                path_data[name]["image"] = os.path.join(self.pres, name, substr, I_name[0])

            # -- path to phantom
            p_name = sorted(glob.glob(os.path.join(self.pres, name, substr, "*phantom*.mat")))
            if p_name:
                path_data[name]["phantom"] = os.path.join(self.pres, name, substr, p_name[0])

        return path_data

    # ------------------------------------------------------------------------------------------------------------------
    def load_OF(self):

        OF = {}

        for name in self.directories.keys():
            fname = self.directories[name]["OF"]
            if fname:
                OF_ = nib.load(fname)
                OF_ = OF_.get_fdata()
                OF[name] = np.array(OF_)
            else:
                OF[name] = fname

        return OF

    # ------------------------------------------------------------------------------------------------------------------
    def load_phantom(self, name):

        scatt = {}

        data = loadmat(name)

        x_min = data['scatt']['x_min']
        x_max = data['scatt']['x_max']

        y_min = data['scatt']['y_min']
        y_max = data['scatt']['y_max']

        z_min = data['scatt']['z_min']
        z_max = data['scatt']['z_max']

        x_scatt = np.array(data['scatt']['x_scatt'])
        y_scatt = np.array(data['scatt']['y_scatt'])
        z_scatt = np.array(data['scatt']['z_scatt'])
        RC_scatt = np.array(data['scatt']['RC_scatt'])


        scatt["x_min"] = x_min
        scatt["x_max"] = x_max
        scatt["y_min"] = y_min
        scatt["y_max"] = y_max
        scatt["z_min"] = z_min
        scatt["z_max"] = z_max

        scatt["x_scatt"] = x_scatt[:,0]
        scatt["y_scatt"] = y_scatt[:,0]
        scatt["z_scatt"] = z_scatt[:,0]
        scatt["RC_scatt"] = RC_scatt[:,0]


        return scatt
    # ------------------------------------------------------------------------------------------------------------------

    def display(self):

        for name in self.directories.keys():
            fname = self.directories[name]["phantom"]
            if fname:
                phantom = self.load_phantom(fname)


                # --- phantom 3D

                # --- Creating figure
                fig = plt.figure(figsize=(10, 7))
                my_cmap = plt.get_cmap('hot')
                ax = plt.axes(projection="3d")

                # Creating plot
                sctt = ax.scatter3D(phantom['x_scatt'] * 10,
                                    phantom['y_scatt'] * 10,
                                    -phantom['z_scatt'] * 10,
                                    c=phantom['RC_scatt'],
                                    s=1,
                                    cmap=my_cmap)

                plt.title("Scatterer map")
                ax.set_xlabel('X position (cm)', fontweight='bold')
                ax.set_ylabel('Y position (cm)', fontweight='bold')
                ax.set_zlabel('Z position (cm)', fontweight='bold')
                fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
                ax.grid(False)
                plt.axis('off')
                ax.set_xlim((-self.image_info['CF'] * self.image_info['width'] * 0.7 * 10, self.image_info['CF']*self.image_info['width'] * 0.7 * 10))
                ax.set_zlim((-self.image_info['CF'] * self.image_info['height'] * 1.3 * 10, self.image_info['CF'] * self.image_info['height'] * 0.2 * 10))
                ax.set_ylim((-self.image_info['depth'] * 10, self.image_info['depth'] * 10))
                ax.view_init(10, 100)

                dir = os.path.join(self.pres, name, 'phantom', '3D_phantom.png')
                plt.savefig(dir, bbox_inches='tight')


                plt.close()

                # --- Creating figure
                plt.figure(figsize=(10, 7))

                plt.scatter(x = phantom['x_scatt'] * 10,
                            y = -phantom['z_scatt'] * 10,
                            c = phantom['RC_scatt'],
                            s = 1,
                            cmap='hot')


                plt.colorbar()
                plt.xlabel('X position (cm)', fontweight='bold')
                plt.ylabel('Z position (cm)', fontweight='bold')
                ax.set_xlim((-self.image_info['CF'] * self.image_info['width'] * 10 / 2, self.image_info['CF'] * self.image_info['width'] * 10 / 2))
                ax.set_ylim((-self.image_info['depth'] * 10, 0))
                plt.title("Scatterer map")

                dir = os.path.join(self.pres, name, 'phantom', '2D_phantom.png')
                plt.savefig(dir, bbox_inches='tight')
                plt.close()


                print(name)

    # ----------------------------------------------------------------------------------------------------------------

    def get_image_info(self):

        fname = list(self.directories.keys())[0]
        test = loadmat(self.directories[fname]['image'])
        info = {}
        info['CF'] = test['image']['CF']
        info['width'] = test['image']['width']
        info['height'] = test['image']['height']
        info['depth'] = 0.1e-2



        return info
    # ------------------------------------------------------------------------------------------------------------------

    def __call__(self):
        self.display()
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Use a breakpoint in the code line below to debug your script.
    pres = "/home/laine/cluster/PROJECTS_IO/SIMULATION/ICCVG"
    displayScatt = displayScatteres(pres)
    displayScatt()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

