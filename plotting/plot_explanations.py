import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


def plot_hamming_heatmap(data, dataset_name, save_path=None):
    heatmap = squareform(pdist(data.apply(lambda row: row < 0.25 * row.min(), axis=1), metric='hamming'))

    # plot
    font = {'family': 'arial',
            'size': 28}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap)

    # axis labels
    if dataset_name.startswith('ex2'):
        plt.xticks(ticks=[0, 2, 6, 10, 18], rotation=0)
        ax.set_xticklabels(['L1', 'I1', 'L2', 'CI', ''])
        label_pos = [0.25, 0.4, 0.4, 0.85, 0]
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            dx = label_pos[i]
            offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)
        plt.xlabel('Fraud cases')

        plt.yticks(ticks=[0, 2, 6, 10, 18], rotation=0)
        ax.set_yticklabels(['L1', 'I1', 'L2', 'CI', ''])
        label_pos = [-0.2, -0.4, -0.4, -0.75, 0]
        for i, label in enumerate(ax.yaxis.get_majorticklabels()):
            dy = label_pos[i]
            offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)
        plt.ylabel('Fraud cases')

    elif dataset_name.startswith('ex1'):
        if 'fraud_2' in dataset_name:
            ticks = [0, 2, 6, 16, 22, 28, 46, 50]
            label_pos_x = [-0.05, 0.225, 0.4, 0.2, 0.3, 0.7, 0.15, 0]
            label_pos_y = [-0.1, -0.255, -0.35, -0.25, -0.25, -0.6, -0.15, 0]
        elif 'fraud_3' in dataset_name:
            ticks = [0, 8, 18, 44, 48, 72, 78, 86]
            label_pos_x = [0.15, 0.25, 0.5, 0.1, 0.5, 0.1, 0.3, 0]
            label_pos_y = [-0.175, -0.225, -0.5, -0.1, -0.5, -0.15, -0.25, 0]
        else:
            raise ValueError("Expected either 'fraud_2' or 'fraud_3' to be specified in variable dataset_name")

        plt.xticks(ticks=ticks, rotation=0)
        ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'I1', 'I2', 'CI', ''])
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            dx = label_pos_x[i]
            offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)
        plt.xlabel('Fraud cases')

        plt.yticks(ticks=ticks, rotation=0)
        ax.set_yticklabels(['L1', 'L2', 'L3', 'L4', 'I1', 'I2', 'CI', ''])
        for i, label in enumerate(ax.yaxis.get_majorticklabels()):
            dy = label_pos_y[i]
            offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)
        plt.ylabel('Fraud cases')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()


if __name__ == '__main__':

    approach = 'AE'
    dataset_name = 'ex1_fraud_2'  # 'ex1_fraud_2', 'ex1_fraud_3', 'ex2_fraud_1'
    save = False

    if dataset_name.startswith('ex2'):
            expl = pd.read_csv(f'../outputs/explanation/ex2/{approach}_shap_{dataset_name}.csv', index_col=0)
    elif dataset_name.startswith('ex1'):
        if 'fraud_2' in dataset_name:
            expl = pd.read_csv(
                f'../outputs/explanation/ex1/{approach}_shap_{dataset_name}.csv',
                index_col=0)
            fraud_order = [27123, 27124, 19919, 19920, 19961, 19962, 13246, 13247, 13248, 13249, 13250, 13251,
                           35015, 35016, 35017, 35018, 660, 661, 662, 32227, 32228, 32229, 17419, 17420, 17421,
                           22709, 22710, 22711, 1207, 4252, 4253, 4254, 4255, 4256, 4257, 1196, 1197, 1198, 1199,
                           1200, 1201, 1202, 1203, 1204, 1205, 1206, 33138, 33139, 33140, 33141]
        elif 'fraud_3' in dataset_name:
            expl = pd.read_csv(
                f'../outputs/explanation/ex1/{approach}_shap_{dataset_name}.csv',
                index_col=0)
            fraud_order = [13759, 13760, 13765, 13766, 22205, 22206, 22211, 22212, 2494, 2495, 2496, 2497, 2498,
                           2499, 2500, 2501, 2502, 2503, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764,
                           8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 35046, 35047,
                           35048, 35049, 684, 685, 5064, 5065, 15745, 15746, 15747, 15748, 15749, 15750, 15751,
                           15752, 15753, 15754, 15755, 15756, 22424, 22425, 22426, 22427, 22428, 22429, 22430,
                           22431, 22432, 22433, 22434, 22435, 361, 362, 1035, 1036, 1037, 1038, 36376, 36377,
                           36378, 36379, 36802, 36803, 36804, 36805]
        else:
            raise ValueError("Expected either 'fraud_2' or 'fraud_3' to be specified in variable dataset_name")

        # Fix order: [Larceny 1 - 4], [Invoice Kickback 1 - 2], [Corporate Injury], then after occurrence
        expl = expl.loc[fraud_order]

        plot_hamming_heatmap(data=expl,
                             dataset_name=dataset_name,
                             save_path=f'./figures/heatmap_{approach}_{dataset_name}.png' if save else None)
