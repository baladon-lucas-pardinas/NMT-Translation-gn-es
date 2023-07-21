import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DATE_COL = 'date'
SCORE_TYPE_COL = 'score_type'
SCORE_COL = 'score'
EPOCH_COL = 'epoch'
MODEL_ID = 'model_name'
DURATION_COL = 'duration'

def plot_metrics_by_epoch(dataframe, metrics, save_path=None):
    num_metrics = len(metrics)
    
    sns.set(style='darkgrid')
    _, axes = plt.subplots(ncols=num_metrics, figsize=(4*num_metrics, 4))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(x='epoch', y=metric, data=dataframe, ax=ax)
        ax.set_title(metric)

        highest_value = dataframe[metric].max()
        max_index = dataframe[dataframe[metric] == highest_value]['epoch'].values[0]
        ax.plot(max_index, highest_value, marker='o', markersize=8, color='red')
        ax.text(0, highest_value, f'Epoch: {int(max_index-1)} \n Max: {highest_value:.2f}', ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=1, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(save_path) if save_path is not None else plt.show()



def plot_metric_by_x_foreach_model(df, metric, x, save_path=None):
    df = df[df[SCORE_TYPE_COL] == metric]
    sns.set(style='darkgrid')
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=x, y=SCORE_COL, hue=MODEL_ID, data=df, errorbar=None)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(save_path) if save_path is not None else plt.show()

def plot_metric_by_epoch_foreach_model(df, metric, save_path=None):
    plot_metric_by_x_foreach_model(df, metric, EPOCH_COL, save_path)

def plot_metric_by_time_foreach_model(df, metric, save_path=None):
    plot_metric_by_x_foreach_model(df, metric, DATE_COL, save_path)

def create_df_from_results_csv(results_csv_path):
    df = pd.read_csv(results_csv_path)
    return df

def plot_max_score_by_model(df: pd.DataFrame, metric, figsize=(20, 6), save_path=None):
    df = df[df[SCORE_TYPE_COL] == metric]
    df = df.groupby(MODEL_ID)[SCORE_COL].max().reset_index()
    df = df.sort_values(by=SCORE_COL, ascending=False)
    sns.set(style='darkgrid')
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=MODEL_ID, y=SCORE_COL, data=df)

    for i in ax.containers:
        ax.bar_label(i,)

    plt.title(metric)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(save_path) if save_path is not None else plt.show()

def plot_time_by_model(df: pd.DataFrame, title: str, duration_col=DURATION_COL, figsize=(20, 6), save_path=None):
    df = df.sort_values(by=duration_col, ascending=False)
    sns.set(style='darkgrid')
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=MODEL_ID, y=duration_col, data=df)

    for i in ax.containers:
        ax.bar_label(i,)

    plt.title(title)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(save_path) if save_path is not None else plt.show()