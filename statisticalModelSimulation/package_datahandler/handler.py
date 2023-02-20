import os
import random
import package_utils.loader         as pul
import numpy                        as np
import matplotlib.pyplot            as plt
import pickle                       as pck
from scipy.stats                    import norm
from scipy.stats                    import rayleigh
from scipy.signal                   import medfilt2d
from scipy.interpolate              import interp1d

# ----------------------------------------------------------------------------------------------------------------------
class handler():

    def __init__(self, p):

        self.path_images = p.PIMAGES
        self.path_pos_lumen = p.PLUMEN
        self.path_pos_interfaces = p.PINTERFACES
        self.path_CF = p.PCF
        self.pres = p.PRES
        self.PIMAGE = p.PRES_IMAGES
        self.PGAUSS = p.PRES_STAT_MODEL
        self.adventicia_depth = 10*1e-3
        self.lumen_distribution = {}
        self.adventicia_distribution = {}
        self.interfaces_distribution = {}
        self.CF = {}
        self.images, self.pos_lumen, self.pos_interfaces = {}, {}, {}

    # ------------------------------------------------------------------------------------------------------------------
    def load_images(self):
        
        fnames = os.listdir(self.path_images)
        fnames.sort()
        for name in fnames:
            self.images[name.split('.')[0]] = pul.load_image(os.path.join(self.path_images, name))

    # ------------------------------------------------------------------------------------------------------------------
    def load_CF(self):

        fnames = os.listdir(self.path_CF)
        for name in fnames:
            with open(os.path.join(self.path_CF, name), 'r') as f:
                cf = f.readlines()
            self.CF[name.split('_CF')[0]] = float(cf[0].split(' ')[0])*1e-3

    # ------------------------------------------------------------------------------------------------------------------
    def load_lumen_pos(self):

        fnames = os.listdir(self.path_pos_lumen)
        fnames.sort()
        bottom_pos, top_pos = fnames[0::2], fnames[1::2]
        for bottom, top in zip(bottom_pos, top_pos):
            patient_name = bottom.split('_')[:-2]
            pname = ''
            for id, key in enumerate(patient_name):
                if id > 0:
                    pname += ("_" + key)
                else:
                    pname += key

            self.pos_lumen[pname] = {}
            self.pos_lumen[pname]["pos_top"] = pul.load_pos(os.path.join(self.path_pos_lumen, top))
            self.pos_lumen[pname]["pos_bottom"] = pul.load_pos(os.path.join(self.path_pos_lumen, bottom))

    # ------------------------------------------------------------------------------------------------------------------
    def load_interfaces_pos(self):

        fnames = os.listdir(self.path_pos_interfaces)
        fnames.sort()
        LI, MA = fnames[0::2], fnames[1::2]
        for LI, MA in zip(LI, MA):
            patient_name = LI.split('_')[:-2]
            pname = ''
            for id, key in enumerate(patient_name):
                if id > 0:
                    pname += ("_" + key)
                else:
                    pname += key
            self.pos_interfaces[pname] = {}
            self.pos_interfaces[pname]["LI"] = pul.load_pos(os.path.join(self.path_pos_interfaces, LI))
            self.pos_interfaces[pname]["MA"] = pul.load_pos(os.path.join(self.path_pos_interfaces, MA))

    # ------------------------------------------------------------------------------------------------------------------
    def image_normalization(self):

        for image in self.images.keys():
            I = self.images[image].copy().astype(float)
            I -= np.min(I)
            I /= np.max(I)
            I *= 255
            I = np.round(I).astype(int)
            self.images[image] = I.copy()

    # ------------------------------------------------------------------------------------------------------------------
    def get_gray_values_lumen(self):

        patches_np = []
        for patient in self.images.keys():

            # --- get central position of the kernel inside the ROI
            # -- do it inside lumen
            x_top, z_top = self.pos_lumen[patient]["pos_top"][0], self.pos_lumen[patient]["pos_top"][1]
            x_bottom, z_bottom = self.pos_lumen[patient]["pos_bottom"][0], self.pos_lumen[patient]["pos_bottom"][1]
            # -- get intersection
            intersection = list(set.intersection(*map(set, [x_top, x_bottom])))
            idx_bottom = np.where((np.asarray(x_bottom) >= intersection[0]) & (np.asarray(x_bottom) <= intersection[-1]))
            idx_top = np.where((np.asarray(x_top) >= intersection[0]) & (np.asarray(x_top) <= intersection[-1]))
            x_top, z_top = np.asarray(x_top)[idx_top], np.asarray(z_top)[idx_top]
            x_bottom, z_bottom = np.asarray(x_bottom)[idx_bottom], np.asarray(z_bottom)[idx_bottom]
            x_top, z_top = list(x_top), list(z_top)
            x_bottom, z_bottom = list(x_bottom), list(z_bottom)
            for id_rel, (id_x, z_start) in enumerate(zip(x_top, z_top)):
                pos_z = int(z_start)
                pos_z_max = int(np.round(z_bottom[id_rel]))
                patches_np.append(list(self.images[patient][pos_z:pos_z_max, id_x]))
        patches_np = [x for sublist in patches_np for x in sublist]
        random.shuffle(patches_np)
        # --- we get the closest square
        self.square = int(np.floor(np.sqrt(len(patches_np))))
        patches_np = np.asarray(patches_np[:self.square**2])
        patches_np = np.reshape(patches_np, (self.square, self.square))
        k_size = 5
        patches_np = np.round(medfilt2d(patches_np, k_size))
        patches_np = patches_np[k_size:-k_size, k_size:-k_size]
        patches_np = patches_np.flatten()
        patches_np = patches_np[np.where(patches_np > 0)[0]]
        self.lumen_distribution['full'] = patches_np

    # ------------------------------------------------------------------------------------------------------------------
    def get_gray_values_IMC(self):

        store_val = {}
        for id in range(101):
            store_val[str(id)] = []
        for patient in list(self.images.keys()):
            non_zeros_LI = np.where(np.asarray(self.pos_interfaces[patient]["LI"][1]) > 0)[0]
            non_zeros_MA = np.where(np.asarray(self.pos_interfaces[patient]["MA"][1]) > 0)[0]
            intersection = list(set.intersection(*map(set, [non_zeros_LI, non_zeros_MA])))
            LI = np.asarray(self.pos_interfaces[patient]["LI"][1])
            MA = np.asarray(self.pos_interfaces[patient]["MA"][1])
            # --- interpolation to be in the range 0/100%
            for id in intersection:
                y_org = self.images[patient][int(LI[id]):int(MA[id]), id]
                if y_org.shape[0] > 5:
                    pos_org = np.linspace(0, 1, y_org.shape[0])
                    pos_q = np.linspace(0, 1, 101)
                    f = interp1d(pos_org, y_org, kind="cubic")
                    y_q = f(pos_q)
                    for pos in range(101):
                        store_val[str(pos)].append(y_q[pos])
        # --- convert to numpy and apply filter
        for key in store_val.keys():
            random.shuffle(store_val[key])
            square = int(np.floor(np.sqrt(len(store_val[key]))))
            arr = np.asarray(store_val[key])
            arr = np.asarray(arr[:square ** 2])
            arr = np.reshape(arr, (square, square))
            k_size = 5
            arr = np.round(medfilt2d(arr, k_size))
            arr = arr[k_size:-k_size, k_size:-k_size]
            store_val[key] = arr.flatten()
        self.interfaces_distribution = store_val

    # ------------------------------------------------------------------------------------------------------------------
    def get_gray_values_avdenticia(self):

        store_val = {}
        for id in range(101):
            store_val[str(id)] = []

        for patient in list(self.images.keys()):
            non_zeros_MA = np.where(np.asarray(self.pos_interfaces[patient]["MA"][1]) > 0)[0]
            MA = np.asarray(self.pos_interfaces[patient]["MA"][1])
            # --- interpolation to be in the range 0/100%
            for id in non_zeros_MA:
                nb_pxl = int(np.round(self.adventicia_depth/self.CF[patient]))
                if (int(MA[id])+nb_pxl) < self.images[patient].shape[0]:
                    y_org = self.images[patient][int(MA[id]):int(MA[id])+nb_pxl, id]

                    if y_org.shape[0] > 5:
                        pos_org = np.linspace(0, 1, y_org.shape[0])
                        pos_q = np.linspace(0, 1, 101)
                        f = interp1d(pos_org, y_org, kind="cubic")
                        y_q = f(pos_q)
                        for pos in range(101):
                            store_val[str(pos)].append(y_q[pos])
        # --- convert to numpy and apply filter
        for key in store_val.keys():
            random.shuffle(store_val[key])
            square = int(np.floor(np.sqrt(len(store_val[key]))))
            arr = np.asarray(store_val[key])
            arr = np.asarray(arr[:square ** 2])
            arr = np.reshape(arr, (square, square))
            k_size = 5
            arr = np.round(medfilt2d(arr, k_size))
            arr = arr[k_size:-k_size, k_size:-k_size]
            store_val[key] = arr.flatten()

        self.adventicia_distribution = store_val

    # ------------------------------------------------------------------------------------------------------------------
    def disp_distribution(self, data: np.ndarray, pres: str, interest: str, resolution: int):

        histo = np.histogram(data, bins=np.arange(0, 256))
        val = histo[0]/float(np.max(histo[0]))
        plt.figure()
        plt.title("Grayscale histogram (lumen).")
        plt.plot(val)
        plt.savefig(os.path.join(pres, 'histogram_' + interest + ".png"))
        plt.close()
        patches = data[:resolution ** 2]
        patches = np.reshape(patches, (resolution, resolution))
        plt.figure()
        plt.imshow(patches, cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(pres, 'distribution_' + interest + ".png"))
        plt.close()
    # ------------------------------------------------------------------------------------------------------------------

    def disp_IMC(self, pres, split):

        knames = list(self.interfaces_distribution.keys())[::split]
        plt.figure()
        plt.title("Grayscale histogram (IMC).")
        for key in knames:
            histo = np.histogram(self.interfaces_distribution[key], bins=np.arange(0, 256))
            val = histo[0] / float(np.max(histo[0]))
            plt.plot(val, label=key)
        plt.legend()
        plt.savefig(os.path.join(pres, "histogram_IMC_several_depth.png"))
        plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.load_lumen_pos()
        self.load_interfaces_pos()
        self.load_CF()
        self.load_images()
        self.image_normalization()
        # --- compute histogram for lumen
        self.get_gray_values_lumen()
        for key in self.lumen_distribution.keys():
            self.disp_distribution(self.lumen_distribution[key], self.PIMAGE, 'lumen', 128)
        fit_rayleigh(self.lumen_distribution, self.PIMAGE, self.PGAUSS, 'lumen')
        # --- compute histogram for adventicia
        self.get_gray_values_avdenticia()
        fit_gaussian(self.adventicia_distribution, self.PIMAGE, self.PGAUSS, 'adventicia', 10)
        # for key in self.adventicia_distribution.keys():
        #     self.disp_distribution(self.adventicia_distribution[key], self.PIMAGE, 'AVD_' + key, 128)
        # --- compute histogram for IMC
        self.get_gray_values_IMC()
        self.disp_IMC(self.PIMAGE, 10)
        fit_gaussian(self.interfaces_distribution, self.PIMAGE, self.PGAUSS, 'IMC', 10)

# ----------------------------------------------------------------------------------------------------------------------
def fit_gaussian(distribution, pres_img, pres_gauss, ROI, split):

    # --- for display only
    knames = list(distribution.keys())[::split]

    plt.figure()
    for key in knames:
        histo = np.histogram(distribution[key], bins=np.arange(0, 256))
        val = histo[0] / float(np.max(histo[0]))
        mu, std = norm.fit(distribution[key])
        x = np.linspace(0, 255, 256)
        p = norm.pdf(x, mu, std)
        p = p / np.max(p)
        p = plt.plot(p, label=key)
        plt.plot(val, c=p[0].get_c())

    plt.legend()
    plt.title("Grayscale histogram (IMC).")
    plt.savefig(os.path.join(pres_img, "fit_gaussian.png"), dpi=1000)
    plt.close()

    gaussian = {}
    for key in distribution.keys():
        histo = np.histogram(distribution[key], bins=np.arange(0, 256))
        val = histo[0] / float(np.max(histo[0]))
        mu, std = norm.fit(distribution[key])
        x = np.linspace(0, 255, 256)
        gaussian[key] = {
            'mu': mu,
            'std': std,
            'bin': [0, 255]
        }

        with open(os.path.join(pres_gauss, ROI + '.pkl'), 'wb') as f:
            pck.dump(gaussian, f)

# ----------------------------------------------------------------------------------------------------------------------
def fit_rayleigh(distribution, pres_img, pres_gauss, ROI):


    rayleigh = {}
    for key in distribution.keys():
        # --- compute scale parameters
        x = np.arange(256)
        mu = np.mean(distribution[key])
        scale_param = np.sqrt((np.sum(np.power(distribution[key]-mu, 2)) - 1/distribution[key].shape[0] * np.sum(distribution[key]-mu) ** 2)/(distribution[key].shape[0]-1))
        y = x / (scale_param ** 2) * np.exp(- x ** 2 / (2 * scale_param ** 2))
        y = y/np.max(y)
        histo = np.histogram(distribution[key], bins=np.arange(0, 256))
        val = histo[0] / float(np.max(histo[0]))
        plt.figure()
        plt.plot(val, label="histogram")
        plt.plot(y, label="distribution")
        plt.savefig(os.path.join(pres_img, "rayleigh_" + ROI + ".png"))

        rayleigh[key] = {"scale_parameters": scale_param}

        with open(os.path.join(pres_gauss, ROI + '.pkl'), 'wb') as f:
            pck.dump(rayleigh, f)