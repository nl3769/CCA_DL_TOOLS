import cv2
from torch.utils.data           import        Dataset
from PIL                        import        Image
import numpy                    as            np
import package_utils.loader     as            pul


# ----------------------------------------------------------------------------------------------------------------------
class FlowDataloader(Dataset):

    def __init__(self):

        self.image_list = []
        self.flow_list = []
        self.CF_list = []
        self.extra_info = []

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):

        # --- modulo, to not be out of the array
        index = index % len(self.image_list)
        # --- read data
        OF = self.read_OF(self.flow_list[index][0])
        I1, I2 = self.read_img(self.image_list[index][0], self.image_list[index][1])
        # --- get the name of the sequence
        name = self.image_list[index][0].split('/')[-1].split('.')[0]

        return I1, I2, OF, name

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_img(pI1, pI2):
        """ Read images and flow from files. """

        I1 = np.mean(np.array(Image.open(pI1)), axis=-1)
        I2 = np.mean(np.array(Image.open(pI2)), axis=-1)
        I1 = cv2.resize(I1, (256, 256), interpolation=cv2.INTER_LINEAR)
        I2 = cv2.resize(I2, (256, 256), interpolation=cv2.INTER_LINEAR)
        I1 = np.expand_dims(I1, axis=0)
        I2 = np.expand_dims(I2, axis=0)

        return I1, I2

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_OF(path):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                OF = np.resize(data, (int(h), int(w), 2))
                z_coef = 256 / OF.shape[0]
                x_coef = 256 / OF.shape[1]
                OF = cv2.resize(OF, (256, 256), interpolation=cv2.INTER_LINEAR)
                OF[..., 0] = OF[..., 0] * x_coef
                OF[..., 1] = OF[..., 1] * z_coef
                OF = np.moveaxis(OF, -1, 0)

                return OF

    # ------------------------------------------------------------------------------------------------------------------
    def __rmul__(self, v):

        self.flow_list  = v * self.flow_list
        self.image_list = v * self.image_list

        return self

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):

        return len(self.image_list)

    # ------------------------------------------------------------------------------------------------------------------
