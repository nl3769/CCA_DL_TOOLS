import os
import sys
import pandas                           as pd
import numpy                            as np
import matplotlib.pyplot                as plt
import seaborn                          as sns
from scipy                              import interpolate
from numba                              import jit
from scipy.spatial.distance             import directed_hausdorff

# ----------------------------------------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def loopPDM(S1, S2, B1, B2):

    DvB1SB2 = []

    for j in range(S1[1]):
        Dvs = []
        for k in range(S2[1] - 1):
            Lambda = np.abs(((B2[1, k+1] - B2[1, k]) * (B1[1, j] - B2[1, k]) + (B2[0, k+1] - B2[0, k]) * (B1[0, j] - B2[0, k])) /
                            ((B2[0, k+1] - B2[0, k]) ** 2 + (B2[1, k+1] - B2[1, k]) ** 2))
            if 0 <= Lambda <= 1:
                Dvs.append(
                    np.abs(((B2[1, k+1] - B2[1, k]) * (-B1[0, j] + B2[0, k]) + (B2[0, k+1] - B2[0, k]) * (B1[1, j] - B2[1, k])) /
                           (np.sqrt((B2[0, k+1] - B2[0, k]) ** 2 + (B2[1, k+1] - B2[1, k]) ** 2))))
            else:
                d1 = np.sqrt((B1[0, j] - B2[0, k]) ** 2 + (B1[1, j] - B2[1, k]) ** 2)
                d2 = np.sqrt((B1[0, j] - B2[0, k+1]) ** 2 + (B1[1, j] - B2[1, k+1]) ** 2)
                Dvs.append(min([d1, d2]))

        DvB1SB2.append(min(Dvs))

    return DvB1SB2

# ----------------------------------------------------------------------------------------------------------------------
def compute_hausdorff_distance(arr1, arr2, CF, common_support, method):

    index, hausdorff_LI, hausdorff_MA = [], [], []

    for patient in common_support.keys():
        if patient in arr1.keys():
            # idx = compute_intersection(arr1[patient]['LI_id'], arr2[patient]['LI_id'], arr1[patient]['MA_id'], arr2[patient]['MA_id'])
            idx = common_support[patient]
            if idx is not None:

                LI_arr1_idx, MA_arr1_idx = get_idx(arr1[patient]['LI_id'], idx), get_idx(arr1[patient]['MA_id'], idx)
                LI_arr2_idx, MA_arr2_idx = get_idx(arr2[patient]['LI_id'], idx), get_idx(arr2[patient]['MA_id'], idx)

                LI1, LI2 = arr1[patient]['LI_val'][LI_arr1_idx], arr2[patient]['LI_val'][LI_arr2_idx]
                LI1, LI2, idx = np.expand_dims(LI1, axis=-1), np.expand_dims(LI2, axis=-1), np.expand_dims(idx, axis=-1)
                LI1, LI2 = np.concatenate((idx, LI1), axis=-1) * CF[patient], np.concatenate((idx, LI2), axis=-1) * CF[patient]

                MA1, MA2 = arr1[patient]['MA_val'][MA_arr1_idx], arr2[patient]['MA_val'][MA_arr2_idx]
                MA1, MA2 = np.expand_dims(MA1, axis=-1), np.expand_dims(MA2, axis=-1)
                MA1, MA2 = np.concatenate((idx, MA1), axis=-1) * CF[patient], np.concatenate((idx, MA2), axis=-1) * CF[patient]

            else:
                LI1, LI2, MA1, MA2 = None, None, None, None

            if LI1 is not None and LI2 is not None and MA1 is not None and MA2 is not None:
                hd_LI = max(directed_hausdorff(LI1, LI2)[0], directed_hausdorff(LI2, LI1)[0])
                hd_MA = max(directed_hausdorff(MA1, MA2)[0], directed_hausdorff(MA2, MA1)[0])

                hausdorff_MA += [{'Value': hd_MA * 1e6, 'Method': method}]
                hausdorff_LI += [{'Value': hd_LI * 1e6, 'Method': method}]
                index.append(patient)

    return hausdorff_LI, hausdorff_MA, index

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def PolyDistMethodHandler(arr, CF, common_support, method):

    index, pdm = [], []

    for patient in common_support.keys():
        if patient in arr.keys():
            idx_0, idx_1 = arr[patient]['LI_id'], arr[patient]['MA_id']
            intersection = list(set.intersection(*map(set, [list(idx_0), list(idx_1)])))
            intersection.sort()
            idx = np.array(intersection)

            if idx is not None:

                LI_arr1_idx, MA_arr1_idx = get_idx(arr[patient]['LI_id'], idx), get_idx(arr[patient]['MA_id'], idx)
                B1, B2 = arr[patient]['LI_val'][LI_arr1_idx], arr[patient]['MA_val'][MA_arr1_idx]
                B1, B2, idx = np.expand_dims(B1, axis=0), np.expand_dims(B2, axis=0), np.expand_dims(idx, axis=0)
                B1, B2 = np.concatenate((idx, B1), axis=0) * CF[patient], np.concatenate((idx, B2), axis=0) * CF[patient]

            else:
                B1, B2 = None, None

            if B1 is not None and B2 is not None:
                PDM, DvB1SB2, DvB2SB1 = PolyDistMethod(B1, B2)
                pdm += [{'Value': PDM*1e6, 'Method': method}]
                index.append(patient)

    return DvB1SB2, DvB2SB1, pdm, index

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def PolyDistMethod(B1, B2):

    # --- get size
    S1, S2 = B1.shape, B2.shape

    if S1[1] > 1 and S2[1] > 1:

        DvB1SB2 = loopPDM(S1, S2, B1, B2)

        Temp = B1
        B1 = B2
        B2 = Temp
        S1, S2 = B1.shape, B2.shape
        DvB2SB1 = loopPDM(S1, S2, B1, B2)
        PDM = np.sum(np.abs(np.array(DvB1SB2))) + np.sum(abs(np.array(DvB2SB1)))
        PDM = PDM / (S1[1] + S2[1])

    else:
        PDM = np.mean(np.abs(B1[1, ] - B2[1, ]))
        DvB1SB2, DvB2SB1 = None, None

    return PDM, DvB1SB2, DvB2SB1

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def enforce_id_integer(pos, val):

    pos_LI_shift = np.roll(pos, 1)
    diff = pos[2:-2] - pos_LI_shift[2:-2]
    count = np.count_nonzero(np.in1d(diff, 1))

    if count != (pos.shape[0]-4):
        pos_target = np.linspace(int(np.ceil(pos[0])), int(np.floor(pos[-1])), int(np.floor(pos[-1]) - np.ceil(pos[0]) + 1))
        f = interpolate.interp1d(pos, val)
        val = f(pos_target)
        pos = pos_target

    pos, val = pos.astype(np.int), val.astype(np.float)

    return pos, val

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_diff(arr1, arr2, CF, common_support, opponent):
    out = []
    index = []
    for patient in list(common_support.keys()):
        if patient in arr2.keys():
            diff = {}
            diff[patient] = {}
            # idx = compute_intersection(arr1[patient]['LI_id'], arr2[patient]['LI_id'], arr1[patient]['MA_id'], arr2[patient]['MA_id']) # if we want to compare on bigger support but there is no sense do to it for fair comparison
            idx = common_support[patient]
            if idx is not None:
                
                LI_arr1_idx, MA_arr1_idx = get_idx(arr1[patient]['LI_id'], idx), get_idx(arr1[patient]['MA_id'], idx)
                LI_arr2_idx, LI_arr2_idx = get_idx(arr2[patient]['LI_id'], idx), get_idx(arr2[patient]['MA_id'], idx)

                diff[patient]['LI'] = np.abs((arr1[patient]['LI_val'][LI_arr1_idx] - arr2[patient]['LI_val'][LI_arr2_idx])) * CF[patient]
                diff[patient]['MA'] = np.abs((arr1[patient]['MA_val'][MA_arr1_idx] - arr2[patient]['MA_val'][LI_arr2_idx])) * CF[patient]
                diff[patient]['IMT'] = np.abs(((arr1[patient]['MA_val'][MA_arr1_idx] - arr1[patient]['LI_val'][LI_arr1_idx]) - (arr2[patient]['MA_val'][LI_arr2_idx] - arr2[patient]['LI_val'][LI_arr2_idx]))) * CF[patient]
            else:
                diff[patient]['IMT'], diff[patient]['MA'], diff[patient]['LI'] = None, None, None

            method_ = opponent.replace('Computerized-', '')
            if diff[patient]['IMT'] is not None:
                for id in range(diff[patient]['IMT'].shape[0]):
                    out += [{'Method': method_, 'Interest': 'IMT', 'Patient': patient, 'Value': diff[patient]['IMT'][id] * 1e6}]
                    index += ['IMT']
                    out += [{'Method': method_, 'Interest': 'LI', 'Patient': patient, 'Value': diff[patient]['LI'][id] * 1e6}]
                    index += ['LI']
                    out += [{'Method': method_, 'Interest': 'MA', 'Patient': patient, 'Value': diff[patient]['MA'][id] * 1e6}]
                    index += ['MA']
            else:
                out += [{'Method': method_, 'Interest': 'IMT', 'Patient': patient, 'Value': None}]
                index += ['IMT']
                out += [{'Method': method_, 'Interest': 'LI', 'Patient': patient, 'Value': None}]
                index += ['LI']
                out += [{'Method': method_, 'Interest': 'MA', 'Patient': patient, 'Value': None}]
                index += ['MA']

    return out, index

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_idx(array, val):

    val = val.flatten()
    array = np.unique(array.flatten())
    idx = np.unique(np.where(np.in1d(array, val))[0])

    return idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_intersection(arr1, arr2, arr3, arr4):

    arr = [list(arr1), list(arr2), list(arr3), list(arr4)]

    intersection = set.intersection(*map(set, arr))
    intersection = list(intersection)
    intersection.sort()
    intersection = np.array(intersection)

    intersection = intersection[10:-10]

    return intersection

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_CF(path):
    # --- get patient names
    CF = {}
    patients = os.listdir(path)
    patients.sort()

    for patient in patients:
        path_ = os.path.join(path, patient)
        with open(path_, 'r') as f:
            CF_val = f.readline()
            CF_val = float(CF_val.split(' ')[0]) * 1e-3  # -> convert in meter
            CF[patient.split('_CF')[0]] = CF_val

    return CF

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_annotation(path, set):

    # --- get patient names
    annotation = {}
    patients = os.listdir(path)

    if set == 'clin':
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname and 'clin' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname and 'clin' in MAname]
    elif set == 'tech':
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname and 'tech' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname and 'tech' in MAname]
    else:
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname]

    # --- ensure that we have the same number of files
    for test in LI_patients:
        if test.replace('LI', 'MA') not in MA_patients:
            print("Problem with file name (fuction get_annotation)")
            sys.exit(1)

    LI_patients.sort()
    MA_patients.sort()
    patients = [patient.split('-')[0] for patient in MA_patients]
    path_contours = []
    for id in range(len(LI_patients)):
        path_contours.append([LI_patients[id], MA_patients[id]])

    for id, pname in enumerate(path_contours):
    # for id, pname in enumerate(path_contours[:30]):
        annotation[patients[id]] = {}
        pLI, pMA = os.path.join(path, pname[0]), os.path.join(path, pname[1])
        with open(pLI, 'r') as f:
            LI = f.readlines()
        with open(pMA, 'r') as f:
            MA = f.readlines()

        if len(LI) == 2 and len(LI) == 2:
            LI_id, MA_id = LI[0].split(' '), MA[0].split(' ')
            LI_id, MA_id = LI_id[:-1], MA_id[:-1]
            LI_val, MA_val = LI[1].split(' ')[:-1], MA[1].split(' ')[:-1]

        else:
            LI, MA = [key.replace(' \n', '') for key in LI], [key.replace(' \n', '') for key in MA]
            LI_val, MA_val = [key.split(' ')[1] for key in LI], [key.split(' ')[1] for key in MA]
            LI_id, MA_id = [key.split(' ')[0] for key in LI], [key.split(' ')[0] for key in MA]

        # --- ensure that indexes are integer, if not interpolation is used
        [pos_LI, val_LI] = enforce_id_integer(np.array(LI_id).astype(np.float), np.array(LI_val).astype(np.float))
        [pos_MA, val_MA] = enforce_id_integer(np.array(MA_id).astype(np.float), np.array(MA_val).astype(np.float))

        annotation[patients[id]]['LI_id'], annotation[patients[id]]['LI_val'] = pos_LI, val_LI
        annotation[patients[id]]['MA_id'], annotation[patients[id]]['MA_val'] = pos_MA, val_MA

    return annotation

# ----------------------------------------------------------------------------------------------------------------------
def mk_cvs_pdm(df_pdm, path, ninfo):

    # --- IMT PDM
    mean_pdm = pd.DataFrame(df_pdm.groupby(['Method'])['Value'].mean())
    std_pdm = pd.DataFrame(df_pdm.groupby(['Method'])['Value'].std())
    mean_pdm.rename(columns={'Value': 'Mean'}, inplace=True)
    std_pdm.rename(columns={'Value': 'Std'}, inplace=True)

    pdm_df = pd.concat([mean_pdm, std_pdm], axis=1)
    pdm_df.to_csv(os.path.join(path, ninfo + '.csv'), index=True, header=True)

# ----------------------------------------------------------------------------------------------------------------------
def mk_cvs_hd(df_hd_LI, df_hd_MA, path, ninfo):


    mean_pdm_LI = pd.DataFrame(df_hd_LI.groupby(['Method'])['Value'].mean())
    std_pdm_LI = pd.DataFrame(df_hd_LI.groupby(['Method'])['Value'].std())
    mean_pdm_LI.rename(columns={'Value': 'Mean HD LI'}, inplace=True)
    std_pdm_LI.rename(columns={'Value': 'Std  HD LI'}, inplace=True)

    mean_pdm_MA = pd.DataFrame(df_hd_MA.groupby(['Method'])['Value'].mean())
    std_pdm_MA = pd.DataFrame(df_hd_MA.groupby(['Method'])['Value'].std())
    mean_pdm_MA.rename(columns={'Value': 'Mean HD MA'}, inplace=True)
    std_pdm_MA.rename(columns={'Value': 'Std  HD MA'}, inplace=True)

    pdm_df = pd.concat([mean_pdm_LI, std_pdm_LI, mean_pdm_MA, std_pdm_MA], axis=1)
    pdm_df.to_csv(os.path.join(path, ninfo + 'hausdorff.csv'), index=True, header=True)
# ----------------------------------------------------------------------------------------------------------------------
def mk_plot_pdm(df, path, ninfo):

    # --- boxplot

    sns.boxplot(df, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_boxplot_full' + '.png'), dpi=500)
    plt.close()

    # -- plot polyline distance
    sns.violinplot(df, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_violinplot_full' + '.png'), dpi=500)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def mk_plot_hausdorff(df_LI, df_MA, path, ninfo):

    # --- boxplot
    sns.boxplot(df_LI, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_boxplot_LI' + '.png'), dpi=500)
    plt.close()

    sns.boxplot(df_MA, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_boxplot_MA' + '.png'), dpi=500)
    plt.close()

    # --- violin plot
    sns.violinplot(df_LI, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_violinplot_LI' + '.png'), dpi=500)
    plt.close()

    sns.violinplot(df_MA, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + '_violinplot_MA' + '.png'), dpi=500)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def mk_cvs_epe(df_diff, path, ninfo):

    mae_IMT = pd.DataFrame(df_diff.filter(like='IMT', axis=0).groupby(['Method'])['Value'].mean())
    mae_IMT.rename(columns={'Value': 'mean IMT'}, inplace=True)
    std_IMT = pd.DataFrame(df_diff.filter(like='IMT', axis=0).groupby(['Method'])['Value'].std())
    std_IMT.rename(columns={'Value': 'std IMT'}, inplace=True)

    mae_LI = pd.DataFrame(df_diff.filter(like='LI', axis=0).groupby(['Method'])['Value'].mean())
    mae_LI.rename(columns={'Value': 'mean LI'}, inplace=True)
    std_LI = pd.DataFrame(df_diff.filter(like='LI', axis=0).groupby(['Method'])['Value'].std())
    std_LI.rename(columns={'Value': 'std LI'}, inplace=True)

    mae_LI = pd.DataFrame(df_diff.filter(like='MA', axis=0).groupby(['Method'])['Value'].mean())
    mae_LI.rename(columns={'Value': 'mean MA'}, inplace=True)
    std_MA = pd.DataFrame(df_diff.filter(like='MA', axis=0).groupby(['Method'])['Value'].std())
    std_MA.rename(columns={'Value': 'std MA'}, inplace=True)

    metrics_df = pd.concat([mae_IMT, mae_LI, mae_LI, std_IMT, std_LI, std_LI], axis=1)
    metrics_df.to_csv(os.path.join(path, ninfo + 'metrics.csv'), index=True, header=True)

# ----------------------------------------------------------------------------------------------------------------------
def mk_plot_epe(df_diff, path, ninfo):

    # --- plot all mae (IMT, LI, MA) on the same graph
    sns.boxplot(df_diff, x='Method', y='Value', showfliers=False, hue='Interest')
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'boxplot_full' + '.png'), dpi=500)
    plt.close()

    # --- plot MA mae
    df = df_diff[df_diff['Interest'] == 'MA']
    sns.boxplot(df, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'boxplot_MA' + '.png'), dpi=500)
    plt.close()

    # --- plot LI mae
    df = df_diff[df_diff['Interest'] == 'LI']
    sns.boxplot(df, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'boxplot_LI' + '.png'), dpi=500)
    plt.close()

    # --- plot IMT mae
    df = df_diff[df_diff['Interest'] == 'IMT']
    sns.boxplot(df, x='Method', y='Value', showfliers=False)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'boxplot_IMT' + '.png'), dpi=500)
    plt.close()

    df_rm_outliers = df_diff[df_diff['Value'] < 600]

    # -- plot all mae (IMT, LI, MA) on the same graph
    sns.violinplot(df_rm_outliers, x='Method', y='Value', showfliers=False, hue='Interest', linewidth=0.2)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'violin_full' + '.png'), dpi=500)
    plt.close()

    # -- plot MA mae
    df = df_rm_outliers[df_rm_outliers['Interest'] == 'MA']
    sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'violin_MA' + '.png'), dpi=500)
    plt.close()

    # -- plot LI mae
    df = df_rm_outliers[df_rm_outliers['Interest'] == 'LI']
    sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'violin_LI' + '.png'), dpi=500)
    plt.close()

    # -- plot IMT mae
    df = df_rm_outliers[df_rm_outliers['Interest'] == 'IMT']
    sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    mean_val = pd.DataFrame(df.groupby(['Method'])['Value'].median())
    min_median = np.min(mean_val.to_numpy().squeeze())
    plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + 'violin_IMT' + '.png'), dpi=500)
    plt.close()