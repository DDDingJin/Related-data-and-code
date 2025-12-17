#%%
import mne
import copy
import glob
import os
import numpy as np
import pandas as pd
import scipy.io as scio
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mne.transforms import apply_trans
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append(r'E:\2025月报')
from dj_read_create import *
from dj_simulation import *
from dj_proj import *
from dj_visual import *
from dj_experiments import *
from dj_metrics import *
matplotlib.use('Qt5Agg')
rng = np.random.RandomState(0)

# 读取MRI信息并创建引导场矩阵
subjects_dir = r'G:\脑子'
subject = subject_name

coregistration_button = 1
if coregistration_button == 1:
    mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir, inst=fif_path, trans=trans_path)
else:
    print('目前无可用的trans_new')


from mne.filter import filter_data

reject_criteria = dict(mag=8e-11)
epoch_tmin = -0.2
epoch_tmax = 0.8
event_ID = 1
baseline_tmin = epoch_tmin
baseline_tmax = 0

for pair_idx, (somato_idx, room_idx) in enumerate(opHFC_pairs):
    somato_file = somato_data[somato_idx]
    room_file = room_data[room_idx]
    trans_name = f'{somato_idx}_{room_idx}'
    trans_path = op.join(r'E:\2025月报\backnoise_test\aud1somato_32_debad', trans_name)
    if os.path.exists(trans_path):
        trans_new = mne.read_trans(trans_path)
        print('加载对应的配准信息文件：', trans_path)
    else:
        trans_new = None
        print('配准信息文件不存在：', trans_path)

    raw_temp = mne.io.read_raw_fif(somato_file, preload=True)
    empty_temp = mne.io.read_raw_fif(room_file, preload=True)

    raw_mix = raw_temp.copy()
    known_noise = empty_temp.copy()
    del raw_temp, empty_temp

    ori_final, _ = extract_ori1pos(raw_mix.info)   # ori_final 规模为 通道数*3
    subjects_dir = r'G:\脑子'
    fwd, subject = extract_fwd(somato_file, patterns, subject_name_mapping, subjects_dir, trans=trans_new)
    leadfield = fwd['sol']['data']

    stim_channel = None
    for ch_name in raw_mix.info['ch_names']:
        if 'STI' in ch_name or 'stim' in ch_name.lower() or 'Trigger' in ch_name:
            stim_channel = ch_name
            print(f"找到刺激通道: {stim_channel}")
            break

    raw_mix = raw_mix.load_data().filter(l_freq=2, h_freq=45, fir_design='firwin')  # 带通滤波
    raw_mix = raw_mix.load_data().notch_filter(freqs=50, notch_widths=1)  # 陷波滤波
    known_noise = known_noise.load_data().filter(l_freq=1, h_freq=48, fir_design='firwin')  # 带通滤波
    known_noise = known_noise.load_data().notch_filter(freqs=50, notch_widths=1)  # 陷波滤波
    raw_mix.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    ori_final, _ = extract_ori1pos(raw_mix.info)
    events = mne.find_events(raw_mix, stim_channel=stim_channel, shortest_event=1)
    epochs_mix = mne.Epochs(raw_mix, events, event_id=event_ID, tmin=epoch_tmin,
                            tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    nepochs = events.shape[0]
    plot_evoked_from_epochs(epochs_mix, n_epochs=nepochs, ylim=1000, seed=None, title='Raw Signal')
    # raw_mix.info['bads'].append('18')
    good_picks = mne.pick_types(raw_mix.info, meg=True, eeg=False, exclude='bads')   # 挑选出除了坏道、刺激通道之外的有效通道

    # 方法执行
    raw_SSP = raw_mix.copy()
    raw_S3P = raw_mix.copy()
    raw_HFC = raw_mix.copy()
    raw_DSSP = raw_mix.copy()
    raw_teHFC = raw_mix.copy()
    # raw_S3P_cluster = raw_mix.copy()
    raw_S3P_own = raw_mix.copy()
    raw_tSSS = raw_mix.copy()

    # 由于10Hz频谱具有较高能量，因此可以被提取 当作噪声处理，在这种Sim1仿真条件下，很容易使得最终的结果为0
    n_fft = 1024 * 2
    n_per_seg = 512 * 2
    n_overlap = 256 * 2
    n_components = 3

    # SSP
    B_clean = apply_SSP(B=raw_SSP._data[good_picks, :], N=known_noise._data[good_picks, :], n_components=3)
    raw_SSP._data[good_picks, :] = B_clean
    del B_clean

    # S3P
    freqs, csd_matrices = compute_csd_matrix_stft(data=known_noise._data[good_picks, :], fmin=2, fmax=45, sfreq=1000,
                                                  n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap)
    S3P_proj, noise_proj, _ = compute_S3P_proj(csd_matrices, n_components=3, test=1)
    B_clean = apply_s3p(data=raw_S3P._data[good_picks, :], projectors=S3P_proj, freqs=freqs, fmin=2, fmax=45,
                        sfreq=1000, method='stft', avoid_ranges=[(0.01, 0.01)], n_fft=n_fft,
                        n_per_seg=n_per_seg, n_overlap=n_overlap)
    raw_S3P._data[good_picks, :] = B_clean[:, :raw_S3P._data.shape[1]]
    del B_clean

    # HFC
    B_clean = apply_HFC(B=raw_HFC._data[good_picks, :], orientations=ori_final.T[good_picks, :])
    raw_HFC._data[good_picks, :] = B_clean
    del B_clean

    # DSSP
    B_clean = apply_DSSP_tuning(F=leadfield[good_picks, :], B=raw_DSSP._data[good_picks, :], n_components=20,
                                   st_correlation=0.998, version=0, vis_data=False, C_components=3)
    raw_DSSP._data[good_picks, :] = B_clean
    del B_clean

    B_clean = apply_teHFC_tuning(B=raw_teHFC._data[good_picks, :], H=known_noise._data[good_picks, :],
                                 orientations=ori_final.T[good_picks, :],
                                 SSP_components=3, st_correlation=0.98, vis_data=False, C_components=1)
    raw_teHFC._data[good_picks, :] = B_clean
    del B_clean

    raw_tSSS = mne.preprocessing.maxwell_filter(raw_tSSS.copy(), origin='auto', int_order=1, ext_order=3, st_duration=None,
                                                st_correlation=0.98, coord_frame='head',
                                                destination=None, regularize='in', ignore_ref=False,
                                                bad_condition='error',
                                                head_pos=None, st_fixed=True,
                                                st_only=False, mag_scale=100.0,
                                                skip_by_annotation=('edge', 'bad_acq_skip'),
                                                extended_proj=(), verbose=None)

    freqs, csd_matrices = compute_csd_matrix_stft(data=raw_S3P_own._data[good_picks, :], fmin=2, fmax=45, sfreq=1000, n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap)
    S3P_proj, noise_proj, _ = compute_S3P_proj(csd_matrices, n_components=n_components, test=1)
    B_clean = apply_s3p(data=raw_S3P_own._data[good_picks, :], projectors=S3P_proj, freqs=freqs, fmin=2, fmax=45, sfreq=1000, method='stft', avoid_ranges=[(0.01, 0.01)], n_fft=n_fft,
                  n_per_seg=n_per_seg, n_overlap=n_overlap)
    raw_S3P_own._data[good_picks, :] = B_clean[:, :raw_S3P_own._data.shape[1]]
    del B_clean

    epochs_SSP = mne.Epochs(raw_SSP, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_S3P = mne.Epochs(raw_S3P, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_HFC = mne.Epochs(raw_HFC, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_DSSP = mne.Epochs(raw_DSSP, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_S3P_own = mne.Epochs(raw_S3P_own, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    # epochs_S3P_cluster = mne.Epochs(raw_S3P_cluster, events, event_id=event_ID, tmin=epoch_tmin,
    #                          tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_teHFC = mne.Epochs(raw_teHFC, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
    epochs_tSSS = mne.Epochs(raw_tSSS, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))

     # nepochs = 100
    ylim = 1000
    # 绘制诱发电位
    highlight_windows = [
        {
            'time_range': (0.1, 0.7),
            'color': 'lightblue',
            'alpha': 0.4,
            'label': 'Target Window'
        }
    ]
    plot_evoked_from_epochs(epochs_mix, n_epochs=nepochs, ylim=ylim, seed=None, title='Band Pass', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_SSP, n_epochs=nepochs, ylim=ylim, seed=None, title='SSP', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_S3P, n_epochs=nepochs, ylim=ylim, seed=None, title='S3P', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_HFC, n_epochs=nepochs, ylim=ylim, seed=None, title='HFC', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_DSSP, n_epochs=nepochs, ylim=ylim, seed=None, title='DSSP', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_S3P_own, n_epochs=nepochs, ylim=ylim, seed=None, title='S3P_own', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_teHFC, n_epochs=nepochs, ylim=ylim, seed=None, title='teHFC', highlight_windows=highlight_windows)
    plot_evoked_from_epochs(epochs_tSSS, n_epochs=nepochs, ylim=ylim, seed=None, title='tSSS', highlight_windows=highlight_windows)

    # 绘制功率谱密度
    raw_SSP.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    raw_S3P.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    raw_HFC.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    raw_DSSP.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    raw_S3P_own.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)
    raw_tSSS.plot_psd(method='welch', fmin=0, fmax=60, tmin=None, tmax=None, dB=True, average=False)

    # 绘制时频图
    reject_criteria = dict(mag=8e-11)
    TF_tmin = epoch_tmin
    TF_tmax = epoch_tmax
    event_ID = 1
    baseline_tmin = epoch_tmin
    baseline_tmax = 0


    from mne.time_frequency import tfr_multitaper
    import numpy as np

    freqs = np.arange(2, 45, 0.1)
    # timefreqs = [(0.05, 4), (0.05, 8), (0.05, 12), (0.1, 4), (0.1, 8), (0.1, 12)]
    timefreqs = [(0.05, 4), (0.05, 8), (0.1, 4), (0.1, 8), (0.1, 12)]
    vmin, vmax = 0, 5e-24

    methods = {
        'mix': epochs_mix,
        'SSP': epochs_SSP,
        'S3P': epochs_S3P,
        'HFC': epochs_HFC,
        'DSSP': epochs_DSSP,
        'S3P_own': epochs_S3P_own,
        'teHFC': epochs_teHFC,
        'tSSS': epochs_tSSS
    }

    # 循环处理每个方法
    for method_name, epochs in methods.items():
        print(f"Processing {method_name}...")
        evoked = epochs.average()
        power_multitaper = tfr_multitaper(evoked, freqs=freqs, n_cycles=2,
                                          time_bandwidth=2.0, use_fft=True,
                                          return_itc=False, decim=3, n_jobs=1)
        power_multitaper.plot_joint(
            baseline=None,
            mode=None,
            timefreqs=timefreqs,
            title=f'{method_name}',
            topomap_args={'vlim': (vmin, vmax),
                          'cmap': 'turbo',
                          'extrapolate': 'head',
                          'border': 'mean',
                          'res': 128,
                          'contours': 7,
                          'sensors': True,
                          'names': None,
                          },
            cmap='turbo',
            vlim=(vmin, vmax)
        )

    # 循环处理每个方法
    for method_name, epochs in methods.items():
        print(f"Processing {method_name}...")
        evoked = epochs.average()
        power_multitaper = tfr_multitaper(evoked, freqs=freqs, n_cycles=2,
                                          time_bandwidth=2.0, use_fft=True,
                                          return_itc=False, decim=3, n_jobs=1)
        power_multitaper.plot_joint(
            baseline=None,  # 不进行基线校正
            mode=None,  # 不使用任何模式转换
            timefreqs=timefreqs,
            topomap_args={'vlim': (vmin, vmax),
                          'cmap': 'RdBu_r',  # 'turbo',
                          'extrapolate': 'head',  # 关键参数：外推到整个头部
                          'border': 'mean',  # 边界处理方式
                          'res': 128,  # 分辨率，数值越大越平滑
                          'contours': 7,
                          'sensors': False,
                          'names': None,
                          'show': False,  # 不显示数值
                          },
            cmap='RdBu_r',  # 'turbo',
            colorbar=False,  # 省略colorbar
            vlim=(vmin, vmax)
        )

#%%  测试4.5 如何融合 ，将得到的U 进行整合  并进行聚类，后提取代表性的成分
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
def sign_normalize_spatial_patterns(spatial_patterns):
    """
    对空间模式进行符号标准化
    让每个向量的最大绝对值元素为正
    """
    normalized_patterns = []
    for pattern in spatial_patterns:
        # 方法1: 让最大绝对值元素为正
        max_abs_idx = np.argmax(np.abs(pattern))
        if pattern[max_abs_idx] < 0:
            pattern = -pattern
        normalized_patterns.append(pattern)

    return np.array(normalized_patterns)
# 使用t-sne k-means 来进行聚类
def tsne_kmeans_clustering(data, n_clusters=5, perplexity=30):
    """
    使用t-SNE降维后进行K-means聚类
    """
    # 1. t-SNE降维
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=1000, learning_rate=200)
    data_tsne = tsne.fit_transform(data)
    # 2. K-means聚类
    print("正在进行K-means聚类...")
    kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42)
    labels_tsne = kmeans_tsne.fit_predict(data_tsne)
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42)
    labels_original = kmeans_original.fit_predict(data)
    # 3. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 原始t-SNE可视化（无聚类）
    axes[0].scatter(data_tsne[:, 0], data_tsne[:, 1],
                    c='gray', alpha=0.6, s=50)
    axes[0].set_title('t-SNE (无聚类标签)')
    axes[0].grid(True, alpha=0.3)
    # t-SNE空间中的K-means聚类
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    for j in range(n_clusters):
        mask = labels_tsne == j
        axes[1].scatter(data_tsne[mask, 0], data_tsne[mask, 1],
                        c=[colors[j]], alpha=0.7, s=50, label=f'Cluster {j}')
    # 添加聚类中心
    centers = kmeans_tsne.cluster_centers_
    axes[1].scatter(centers[:, 0], centers[:, 1],
                    c='red', marker='x', s=200, linewidths=3, label='Centers')
    axes[1].set_title('t-SNE + K-means')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # 原始空间K-means在t-SNE中的可视化
    for j in range(n_clusters):
        mask = labels_original == j
        axes[2].scatter(data_tsne[mask, 0], data_tsne[mask, 1],
                        c=[colors[j]], alpha=0.7, s=50, label=f'Cluster {j}')
    axes[2].set_title('K-means (原始空间) 在t-SNE中的可视化')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    results = {
        'tsne_kmeans': {
            'labels': labels_tsne,
            'centers': centers
        },
        'original_kmeans': {
            'labels': labels_original,
        },
        'tsne_embedding': data_tsne
    }
    return results
# 提取代表性样本
def extract_representative_patterns(classified_data, method='closest_to_center'):
    """精简版：从聚类中提取代表性样本"""
    spatial_patterns = []
    pattern_info = {}

    for cluster_id, cluster_data in classified_data.items():
        if len(cluster_data) == 0:
            continue

        if len(cluster_data) == 1:
            representative = cluster_data[0]
            selected_idx = 0
        else:
            if method == 'closest_to_center':
                center = np.mean(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - center, axis=1)
                selected_idx = np.argmin(distances)

            elif method == 'medoid':
                # 计算每个样本到其他所有样本的距离之和
                distances_sum = np.array([
                    np.sum([np.linalg.norm(cluster_data[i] - cluster_data[j])
                            for j in range(len(cluster_data)) if i != j])
                    for i in range(len(cluster_data))
                ])
                selected_idx = np.argmin(distances_sum)

            elif method == 'highest_energy':
                energies = np.linalg.norm(cluster_data, axis=1)
                selected_idx = np.argmax(energies)

            representative = cluster_data[selected_idx]

        spatial_patterns.append(representative)
        pattern_info[cluster_id] = {
            'n_samples': len(cluster_data),
            'selected_idx': selected_idx
        }

    return np.array(spatial_patterns), pattern_info


raw_1 = raw_mix.copy()
# 由于10Hz频谱具有较高能量，因此可以被提取 当作噪声处理，在这种Sim1仿真条件下，很容易使得最终的结果为0
n_fft = 1024 * 2
n_per_seg = 512 * 2
n_overlap = 256 * 2
n_components = 4

freqs, csd_matrices = compute_csd_matrix_stft(data=raw_1._data[good_picks, :], fmin=2, fmax=45, sfreq=1000,
                                              n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, use_real_only=True)
S3P_proj, noise_proj, U_stft = compute_S3P_proj(csd_matrices, n_components=n_components, test=1)

all_spatial_bases = []
unique_labels = []
if isinstance(U_stft, list):
    global_idx = 0
    for window_idx, U_window in enumerate(U_stft):   # 处理符号问题
        print(f"处理频率点 {window_idx}")
        spatial_bases_window = U_window.T  # 转置后每行是一个空间模式
        spatial_bases_window = sign_normalize_spatial_patterns(spatial_bases_window)
        all_spatial_bases.append(spatial_bases_window)
        n_components = spatial_bases_window.shape[0]
        for comp_idx in range(n_components):
            unique_labels.append(global_idx)
            global_idx += 1

all_spatial_bases = np.vstack(all_spatial_bases)

# 执行简化的t-SNE + K-means聚类分析
n_clusters = 10
tsne_results = tsne_kmeans_clustering(all_spatial_bases, n_clusters=n_clusters)

cluster_labels = tsne_results['tsne_kmeans']['labels']  # 这是0-4的标签
optimal_k = n_clusters  # 或者你可以设置为 len(np.unique(cluster_labels))

# 根据聚类结果分类数据
classified_data = {}
classified_labels = {}
for cluster_id in range(1, optimal_k + 1):  # 1, 2, 3, 4, 5
    # 找到属于当前聚类的数据点（注意：cluster_labels是0-4，我们要映射到1-5）
    mask = cluster_labels == (cluster_id - 1)  # 0对应类别1，1对应类别2，以此类推
    classified_data[cluster_id] = all_spatial_bases[mask]
    # 这里需要修改：直接使用mask来获取对应的索引
    classified_labels[cluster_id] = np.where(mask)[0]  # 获取满足条件的索引

raw = raw_mix.copy()
# 可视化代表性的向量
patterns, info = extract_representative_patterns(classified_data, method='closest_to_center')
S = range(1, len(patterns) + 1)
U = patterns.T
n_components = 30
vis_noise_U(U, S, raw.info, n_components=n_components, figsize=(15, 10),
            title=f"Components Analysis ({len(patterns)} samples)", max_cols=6)

from scipy.linalg import qr
Q, R = qr(U, mode='economic')
raw_1 = raw_mix.copy()
raw_1_noise = raw_mix.copy()

n_index = 8
U_temp = Q[:, :n_index]  # 保留前9列
identity_matrix = np.identity(len(U))
raw_1._data[good_picks, :] = (identity_matrix - U_temp @ U_temp.T) @ raw_1._data[good_picks, :]
raw_1_noise._data[good_picks, :] = (U_temp @ U_temp.T) @ raw_1_noise._data[good_picks, :]

B_clean = apply_CTSP_tuning(data_int=raw_1._data[good_picks, :], data_res=raw_1_noise._data[good_picks, :], corr=0.9, C_components=1) # 0/1都行
raw_1._data[good_picks, :] = B_clean
del B_clean

epochs_1 = mne.Epochs(raw_1, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
plot_evoked_from_epochs(epochs_1, n_epochs=nepochs, ylim=ylim, seed=None, title=f'proj_{n_index}')

epochs_1_noise = mne.Epochs(raw_1_noise, events, event_id=event_ID, tmin=epoch_tmin,
                             tmax=epoch_tmax, reject=reject_criteria, baseline=(baseline_tmin, baseline_tmax))
plot_evoked_from_epochs(epochs_1_noise, n_epochs=nepochs, ylim=ylim, seed=None, title=f'proj_{n_index}_noise')



evoked = epochs_1.average()
freqs = np.arange(2, 42, 0.1)
power_multitaper = tfr_multitaper(evoked, freqs=freqs, n_cycles=2,
                                  time_bandwidth=2.0, use_fft=True,
                                  return_itc=False, decim=3, n_jobs=1)
# timefreqs = [(0.05, 4), (0.05, 8), (0.05, 12), (0.1, 4), (0.1, 8), (0.1, 12)]
timefreqs = [(0.05, 4), (0.05, 8), (0.1, 4), (0.1, 8), (0.1, 12)]
# 在绘制前设置matplotlib的默认图像大小
plt.rcParams['figure.figsize'] = [8, 6]  # 设置默认图像大小
power_multitaper.plot_joint(
    baseline=None,          # 不进行基线校正
    mode=None,              # 不使用任何模式转换
    timefreqs=timefreqs,
    topomap_args={'vlim': (vmin, vmax),
                  # 'cmap': 'RdBu_r',# 'turbo',
                  'cmap': 'turbo',# 'turbo',
                  'extrapolate': 'head',  # 关键参数：外推到整个头部
                  'border': 'mean',  # 边界处理方式
                  'res': 128,  # 分辨率，数值越大越平滑
                  'contours': 7,
                  'sensors': False,
                  'names': None,
                  'show': False,  # 不显示数值
                  },
    # cmap='RdBu_r',#'turbo',
    cmap='turbo',
    colorbar=False,  # 省略colorbar
    vlim=(vmin, vmax)
)
for ax in fig.get_axes():
    # 移除x轴和y轴的刻度标签（数字）
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 移除x轴和y轴的标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 可选：也可以完全移除刻度线
    # ax.set_xticks([])
    # ax.set_yticks([])

plt.tight_layout()
plt.show()


def dj_source_snr_vis(epochs, evoked, fwd, subjects_dir, tmin=-0.2, tmax=0.0, inverse_method="MNE", save_path=None):
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method="shrunk", verbose=True)
    # rank_info = mne.compute_rank(evoked.info, tol=1e-4, tol_kind='relative')
    inv_op = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, rank=None, fixed=True, verbose=True)
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2, inverse_method, verbose=True)
    snr_stc = stc.estimate_snr(evoked.info, fwd, cov)
    snr_stc.data = snr_stc.data
    ave = np.mean(snr_stc.data, axis=0)

    # 自适应确定lims - 使用稳健的百分位数方法
    low_lim = np.percentile(snr_stc.data, 15)  # 15%分位数作为下限
    mid_lim = np.percentile(snr_stc.data, 70)  # 70%分位数作为中值
    high_lim = np.percentile(snr_stc.data, 90)  # 95%分位数作为上限
    lims_auto = (low_lim, mid_lim, high_lim)
    print('推荐的SNR范围为：', lims_auto)
    lims_auto = (np.float64(-110), np.float64(-85), np.float64(-55))
    lims_auto = (np.float64(-100), np.float64(-80), np.float64(-50))
    # fig, ax = plt.subplots(layout="constrained")
    # ax.plot(evoked.times, ave)
    # ax.set(xlabel="Time (s)", ylabel="SNR")

    time_start, time_end = 0.045, 0.057
    time_start, time_end = 0.056, 0.056
    time_mask = (evoked.times >= time_start) & (evoked.times <= time_end)
    masked_ave = np.full_like(ave, -np.inf)
    masked_ave[time_mask] = ave[time_mask]
    maxidx = np.argmax(masked_ave)

    snr_at_timepoint = snr_stc.data[:, maxidx] # 获取该时间点的SNR值
    # top_indices = np.argsort(snr_at_timepoint)[-n_voxels:]
    # top_snr_values = snr_at_timepoint[top_indices]  # 获取这些体素的SNR值（保持原始符号）

    kwargs = dict(
        initial_time=evoked.times[maxidx],
        hemi="both",
        views='lateral',  # 或者用自定义视角
        # surface='inflated',
        # surface='pial',
        surface='white',
        subjects_dir=subjects_dir,
        background='white',
        alpha=0.8,
        size=(800, 600),  # 稍微大一点
        clim=dict(kind="value", lims=lims_auto),
        transparent=True,
        colormap="RdBu_r",  # 红蓝对比色
        # colormap='Spectral_r',
    # show_traces=False,
        time_viewer=False
    )

    brain = snr_stc.plot(**kwargs)
    brain.show_view(azimuth=180, elevation=40, distance=400)
    brain._renderer.plotter.remove_all_lights(only_active=False)

    if save_path:
        # 确保路径以.tif结尾
        if not save_path.endswith('.tif') and not save_path.endswith('.tiff'):
            save_path += '.tif'

        # 保存为TIFF格式
        brain.save_image(save_path)
        print(f"Brain plot saved as: {save_path}")
    return snr_at_timepoint
snr_stc = dj_source_snr_vis(epochs_1, evoked_1, fwd, subjects_dir, tmin=tmin, tmax=tmax, inverse_method=inverse_method, save_path=None)


#%%  计算信噪比
evoked_mix = epochs_mix.average()
evoked_S3P_cluster = epochs_1.average()
evoked_SSP = epochs_SSP.average()
evoked_HFC = epochs_HFC.average()
evoked_DSSP = epochs_DSSP.average()
evoked_S3P = epochs_S3P.average()
evoked_tSSS = epochs_tSSS.average()
evoked_teHFC = epochs_teHFC.average()
parts = [
    (evoked_mix, "Raw"),
    (evoked_SSP, "SSP"),
    (evoked_HFC, "HFC"),
    (evoked_DSSP, "DSSP"),
    (evoked_S3P, "S3P"),
    (evoked_tSSS, "tSSS"),
    (evoked_teHFC, "teHFC"),
    # (evoked_S3P_own, "S3P_own"),
    (evoked_S3P_cluster, "S3P_cluster"),
]
for evoked, method_name in parts:

    good_channels = mne.pick_types(evoked.info, meg=True, eeg=False, exclude='bads')
    evoked_data = evoked._data
    times = evoked.times

    ER_SNR = dj_calculate_ER_SNR(evoked_data, times, good_channels, baseline_mode='exclude_signal',
                                 signal_window=[0.0, 0.1], global_method='mean', top_percent=1)
    print(f'{method_name}计算得到的信噪比为：{round(ER_SNR, 2)}')

#%% 计算SF
good_channels = mne.pick_types(raw_mix.info, meg=True, eeg=False, exclude='bads')
raw_S3P_cluster = raw_1.copy()
epochs_S3P_cluster = epochs_1.copy()
parts = [
    (raw_SSP, "SSP"),
    (raw_HFC, "HFC"),
    (raw_DSSP, "DSSP"),
    (raw_S3P, "S3P"),
    (raw_tSSS, "tSSS"),
    (raw_teHFC, "teHFC"),
    (raw_S3P_cluster, "S3P_cluster"),
]

for raw, method_name in parts:
    sf1 = calculate_SF(B_before=raw_mix._data[good_channels, :], B_after=raw._data[good_channels, :], specific_freq=None, mode='real')

    print(f'{method_name}计算得到的SF为：{round(sf1, 2)}')


#%%
# 将SNR映射到白质 使用 绝对值来映射
def dj_source_snr_vis(epochs, evoked, fwd, subjects_dir, tmin=-0.2, tmax=0.0, inverse_method="MNE", save_path=None):
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method="shrunk", verbose=True)
    # rank_info = mne.compute_rank(evoked.info, tol=1e-4, tol_kind='relative')
    inv_op = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, rank=None, fixed=True, verbose=True)
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2, inverse_method, verbose=True)
    snr_stc = stc.estimate_snr(evoked.info, fwd, cov)
    snr_stc.data = snr_stc.data
    ave = np.mean(snr_stc.data, axis=0)

    # 自适应确定lims - 使用稳健的百分位数方法
    low_lim = np.percentile(snr_stc.data, 15)  # 15%分位数作为下限
    mid_lim = np.percentile(snr_stc.data, 70)  # 70%分位数作为中值
    high_lim = np.percentile(snr_stc.data, 90)  # 95%分位数作为上限
    lims_auto = (low_lim, mid_lim, high_lim)
    print('推荐的SNR范围为：', lims_auto)
    lims_auto = (np.float64(-110), np.float64(-85), np.float64(-55))
    lims_auto = (np.float64(-100), np.float64(-80), np.float64(-50))
    # fig, ax = plt.subplots(layout="constrained")
    # ax.plot(evoked.times, ave)
    # ax.set(xlabel="Time (s)", ylabel="SNR")

    time_start, time_end = 0.045, 0.057
    time_start, time_end = 0.056, 0.056
    time_mask = (evoked.times >= time_start) & (evoked.times <= time_end)
    masked_ave = np.full_like(ave, -np.inf)
    masked_ave[time_mask] = ave[time_mask]
    maxidx = np.argmax(masked_ave)

    snr_at_timepoint = snr_stc.data[:, maxidx] # 获取该时间点的SNR值

    kwargs = dict(
        initial_time=evoked.times[maxidx],
        hemi="both",
        views='lateral',  # 或者用自定义视角
        # surface='inflated',
        # surface='pial',
        surface='white',
        subjects_dir=subjects_dir,
        background='white',
        alpha=0.8,
        size=(800, 600),  # 稍微大一点
        clim=dict(kind="value", lims=lims_auto),
        transparent=True,
        colormap="RdBu_r",  # 红蓝对比色
    # show_traces=False,
        time_viewer=False
    )

    brain = snr_stc.plot(**kwargs)
    brain.show_view(azimuth=180, elevation=40, distance=400)
    brain._renderer.plotter.remove_all_lights(only_active=False)

    if save_path:
        # 确保路径以.tif结尾
        if not save_path.endswith('.tif') and not save_path.endswith('.tiff'):
            save_path += '.tif'

        # 保存为TIFF格式
        brain.save_image(save_path)
        print(f"Brain plot saved as: {save_path}")
    return snr_at_timepoint

inverse_method = "MNE"
tmin, tmax = -0.2, 0.0
# dj_source_snr_vis(epochs_mix, epochs_mix.average(), fwd, subjects_dir, tmin=tmin, tmax=tmax, inverse_method=inverse_method)

# 根据parts变量修改的循环代码
parts_epochs = [
    (epochs_mix, evoked_mix, "mix"),
    (epochs_SSP, evoked_SSP, "SSP"),
    (epochs_HFC, evoked_HFC, "HFC"),
    (epochs_DSSP, evoked_DSSP, "DSSP"),
    (epochs_S3P, evoked_S3P, "S3P"),
    (epochs_tSSS, evoked_tSSS, "tSSS"),
    (epochs_teHFC, evoked_teHFC, "teHFC"),
    (epochs_S3P_cluster, evoked_S3P_cluster, "S3P_cluster"),
]

# 循环处理每个方法
snr_results = {}
for epochs, evoked, method_name in parts_epochs:
    print(f"Processing {method_name}...")
    # save_path = f"E:/2025月报/backnoise_test/meeting/SNR_brain_{method_name}_{inverse_method}.tif"
    save_path = None
    # save_path = r"E:\2025月报\backnoise_test\meeting_pro\0"
    # save_path = rf"E:\2025月报\backnoise_test\meeting_pro\{method_name}.tif"
    snr_stc = dj_source_snr_vis(epochs, evoked, fwd, subjects_dir, tmin=tmin, tmax=tmax, inverse_method=inverse_method, save_path=save_path)
    snr_results[method_name] = snr_stc
