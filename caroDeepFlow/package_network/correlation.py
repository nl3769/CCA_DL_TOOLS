import torch
import torch.nn.functional                  as F
import package_network.grid_handler         as gridHandler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

# ----------------------------------------------------------------------------------------------------------------------
class CorrBlock:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, fmap1=None, fmap2=None, num_levels=4, radius=4, m_name=''):

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.m_name = m_name

        if fmap1 is not None:

            # --- all pairs correlation
            corr = CorrBlock.corr(fmap1, fmap2)

            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

            self.corr_pyramid.append(corr)
            # --- on fait un average pooling de 2 (on réduit par 4 la dimension à chaque itération)
            for i in range(self.num_levels - 1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                self.corr_pyramid.append(corr)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, coords):

        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        # print(f'{self.num_levels = }')
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # on récupère la correlation associé à l'étage de la pyramide

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)  # points en x
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)  # points en y
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'),
                                axis=-1)  # on génère la grille (delta.shape = (2*r+1, 2*r+1, 2))

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1,
                                          2) / 2 ** i  # les coordonnées de l'image -> on divise par 2^i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            coords_lvl = torch.vstack((coords_lvl, coords_lvl)) if '3T' in self.m_name else coords_lvl
            corr = gridHandler.bilinear_sampler(corr, coords_lvl)

            # print(f'{corr.shape = }')
            corr = corr.view(batch, h1, w1, -1)
            # print(f'{corr.shape = }')
            out_pyramid.append(corr)
            # print(f'{corr.shape = }')

        # for i in range(len(out_pyramid)):
        #     print(f'{out_pyramid[i].shape = }')

        out = torch.cat(out_pyramid, dim=-1)  # on concatène tout sur la dernière dimension

        return out.permute(0, 3, 1, 2).contiguous().float()

    # ------------------------------------------------------------------------------------------------------------------
    def cat_correlation_blocks(self, corr_01, corr_12):

        for level in range(self.num_levels):
            self.corr_pyramid.append(torch.cat((corr_01.corr_pyramid[level], corr_12.corr_pyramid[level]), dim=0))

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)  # il redimensionne les features
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # permet de faire le produit terme à à terme
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
class AlternateCorrBlock:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)

        return corr / torch.sqrt(torch.tensor(dim).float())

# ----------------------------------------------------------------------------------------------------------------------
