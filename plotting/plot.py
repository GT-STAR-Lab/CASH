import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_dataframe(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")

def load_dataframe(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    else:
        print(f"File {filename} not found.")
        return None

def fetch_wandb_data(project_name, tags, metrics, agent_param_count_metric):
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, {"tags": {"$in": tags}})
    print(f"found {len(runs)} runs with tags {tags}")
    
    all_run_data = []
    for run in runs:
        print(run.name)
        print(run.config)

        # QMIX/DAgger
        # hyperaware = run.config.get("alg")["AGENT_HYPERAWARE"]
        # cap_aware = run.config.get("env")["ENV_KWARGS"]["capability_aware"]
        # init_scale = run.config.get("alg")["AGENT_HYPERNET_KWARGS"]["INIT_SCALE"]
        # use_ln = run.config.get("alg")["AGENT_HYPERNET_KWARGS"]["USE_LAYER_NORM"]

        # MAPPO
        hyperaware = run.config.get("AGENT_HYPERAWARE")
        cap_aware = run.config.get("ENV_KWARGS")["capability_aware"]
        init_scale = run.config.get("AGENT_HYPERNET_KWARGS")["INIT_SCALE"]
        use_ln = run.config.get("AGENT_HYPERNET_KWARGS")["USE_LAYER_NORM"]

        # drop the one CASH run with init_scale = 0.5!
        if (hyperaware and cap_aware) and init_scale != 0.2:
            print(f"\nDropping {run.name} w/ init_scale != 0.2!\n")
            continue

        # NOTE: agent_param_count is only plotted once and thus cannot be found in the same scan_history() call as the other metrics, finding manually here:
        agent_param_count = run.scan_history(keys=[agent_param_count_metric])
        agent_param_count = next(agent_param_count)[agent_param_count_metric]
        print('agent_param_count', agent_param_count)

        # get metrics of run over time
        history = run.scan_history(keys=metrics)

        run_data = pd.DataFrame(history)
        run_data['run_id'] = run.id
        run_data['run_name'] = run.name
        run_data['tags'] = ', '.join(run.tags)
        run_data['hyperaware'] = hyperaware
        run_data['cap_aware'] = cap_aware
        run_data['use_ln'] = use_ln
        run_data['agent_param_count'] = agent_param_count

        # for DAgger I didn't log TS, add that (hardcoded based on manual math...)
        # run_data['timestep'] = 10 * run_data['policy/updates']

        print(run_data.head())

        all_run_data.append(run_data)
    
    return pd.concat(all_run_data, ignore_index=True)

def baseline_name(row):
    if row['hyperaware']: 
        if row['cap_aware']:
            return "CASH"
    else:
        if row['cap_aware']:
            return "RNN-EXP"
        else:
            return "RNN-IMP"

def get_from_wandb():
    # get runs from wandb
    # NOTE: I've been hardcoding the right things in...

    project_name = "JaxMARL"

    # tags = ['final-qmix-fire']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['final-qmix-hmt']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['final-mappo-fire']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'actor_param_count'

    # tags = ['final-mappo-hmt']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    # agent_param_count_metric = 'actor_param_count'

    # tags = ['final-dagger-fire']
    # fire_metrics = ['policy/updates', 'policy/returns', 'policy/fire_success_rate', 'policy/snd', 'policy/pct_fires_put_out']
    # agent_param_count_metric = 'policy/agent_param_count'

    # tags = ['final-dagger-hmt']
    # dagger_hmt_metrics = ['policy/updates', 'policy/returns', 'policy/snd', 'policy/makespan', 'policy/quota_met', 'policy/loss']
    # agent_param_count_metric = 'policy/agent_param_count'

    # tags = ['final-dagger-hmt-ln']
    # dagger_hmt_metrics = ['policy/updates', 'policy/returns']
    # agent_param_count_metric = 'policy/agent_param_count'

    # tags = ['ln-ablation-qmix-fire']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['ln-ablation-qmix-transport']
    # metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['ln-ablation-mappo-fire']
    # metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'actor_param_count'

    tags = ['ln-ablation-mappo-transport']
    metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    agent_param_count_metric = 'actor_param_count'

    df = fetch_wandb_data(project_name, tags, metrics, agent_param_count_metric)
    filename = f"{tags[0]}.pkl" 
    save_dataframe(df, filename)

    print("saved")
    print(df.head())

def smooth_and_downsample(df, y_column, mean_window=50, std_window=50, downsample_factor=10):
    """
    Creates a new dataframe with smoothed and downsampled data, with separate
    smoothing controls for mean and standard deviation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_column : str
        Column to analyze
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    smoothed_data = []
    df_copy = df.copy()
    
    for baseline in df_copy['baseline'].unique():
        baseline_data = df_copy[df_copy['baseline'] == baseline].copy()
        baseline_data = baseline_data.sort_values('timestep')
        
        # Group by timestep to calculate mean and std
        grouped = baseline_data.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()
        
        # Smooth mean and std separately
        grouped['smooth_mean'] = grouped['mean'].rolling(
            window=mean_window, min_periods=1, center=True).mean()
        grouped['smooth_std'] = grouped['std'].rolling(
            window=std_window, min_periods=1, center=True).mean()
        
        # Downsample
        grouped = grouped.iloc[::downsample_factor]
        
        # Create dataframe with smoothed mean and smoothed std
        smoothed_df = pd.DataFrame({
            'timestep': grouped['timestep'],
            f'{y_column}': grouped['smooth_mean'],
            f'{y_column}_std': grouped['smooth_std'],
            'baseline': baseline
        })
        
        smoothed_data.append(smoothed_df)
    
    return pd.concat(smoothed_data)

def plot_metrics(df, y_label, y_column, title, mean_window, std_window, downsample_factor, save_folder):
    """
    Generic plotting function for any metric with separate smoothing controls
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    y_column : str
        The name of the column to plot on y-axis
    title : str
        The title of the plot
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    # Create smoothed version of the data for plotting
    smoothed_df = smooth_and_downsample(df, y_column=y_column, 
                                      mean_window=mean_window,
                                      std_window=std_window,
                                      downsample_factor=downsample_factor)
    
    # plt.figure(figsize=(6.4, 4.8))
    plt.figure(figsize=(4, 3))
    
    base_palette = sns.color_palette()
    palette = {
        'CASH': base_palette[2],
        'RNN-EXP': base_palette[0],
        'RNN-IMP': base_palette[1],
        'CASH w/o LN': base_palette[3],
    }
    
    # Plot each baseline separately
    for baseline in smoothed_df['baseline'].unique():
        baseline_data = smoothed_df[smoothed_df['baseline'] == baseline]
        color = palette[baseline]
        
        # Plot mean line
        plt.plot(baseline_data['timestep'], baseline_data[y_column], 
                color=color, label=baseline, linewidth=2)
        
        # Add error bands with smoothed std
        plt.fill_between(baseline_data['timestep'],
                        baseline_data[y_column] - baseline_data[f'{y_column}_std'],
                        baseline_data[y_column] + baseline_data[f'{y_column}_std'],
                        color=color, alpha=0.2)
    
    plt.xlabel('Timestep')
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend(loc='best')
    plt.tight_layout(pad=0.5)

    # plt.show()
    plt.savefig(f"{save_folder}/{title.replace('/', '')}-{y_label}.png".lower().replace(' ', '-'))

def plot_from_saved():
    """
    # for legend
    filename = "final-qmix-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder=".")
    exit(0)
    """

    # QMIX Fire
    filename = "final-qmix-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)
    # Fire
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='SND', y_column='test_snd', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Success Rate', y_column='test_fire_success_rate', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Pct of Fires Extinguished', y_column='test_pct_fires_put_out', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")

    # qmix hmt
    filename = "final-qmix-hmt.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)

    # HMT
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='SND', y_column='test_snd', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Success Rate', y_column='test_quota_met', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")
    plot_metrics(df, y_label='Makespan', y_column='test_makespan', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="qmix")

    # mappo Fire
    filename = "final-mappo-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)
    # Fire
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='SND', y_column='test_snd', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Success Rate', y_column='test_fire_success_rate', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Pct of Fires Extinguished', y_column='test_pct_fires_put_out', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")

    # mappo HMT
    filename = "final-mappo-hmt.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)

    # HMT
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='SND', y_column='test_snd', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Success Rate', y_column='test_quota_met', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")
    plot_metrics(df, y_label='Makespan', y_column='test_makespan', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=10, save_folder="mappo")


    # DAGGER FIRE
    # load preprocessed data
    filename = "final-dagger-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)
    # Fire
    plot_metrics(df, y_label='Training Returns', y_column='policy/returns', title='DAgger / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='SND', y_column='policy/snd', title='DAgger / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='Success Rate', y_column='policy/fire_success_rate', title='DAgger / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='Pct of Fires Extinguished', y_column='policy/pct_fires_put_out', title='DAgger / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")

    # dagger HMT
    filename = "final-dagger-hmt.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)

    # HMT
    plot_metrics(df, y_label='Training Returns', y_column='policy/returns', title='DAgger / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='SND', y_column='policy/snd', title='DAgger / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='Success Rate', y_column='policy/quota_met', title='DAgger / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")
    plot_metrics(df, y_label='Makespan', y_column='policy/makespan', title='DAgger / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder="dagger")

def plot_parameter_counts(param_dict, title):
    """
    Creates a horizontal bar chart showing parameter counts for different architectures
    """
    plt.figure(figsize=(4, 1))
    
    # Data
    baselines = list(param_dict.keys())
    params = list(param_dict.values())
    
    # Colors
    base_palette = sns.color_palette()
    color_dict = {
        'CASH': base_palette[2],
        'RNN-EXP': base_palette[0],
        'RNN-IMP': base_palette[1]
    }
    colors = [color_dict[baseline] for baseline in baselines]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(baselines))
    bars = plt.barh(y_pos, params, color=colors)
    
    # Customize the plot
    # plt.xlabel('Learnable Parameters')
    # plt.ylabel('Architecture')
    plt.title(title)
    
    # Format x-axis to use scientific notation
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # Or alternatively, divide by 10000 manually:
    # plt.xticks(plt.xticks()[0], [f'{x/1e4:.1f}' for x in plt.xticks()[0]])
    
    # Add value labels inside the bars, aligned to the right end
    # for i, bar in enumerate(bars):
    #     width = bar.get_width()
    #     plt.text(width * 0.95,  # Position at 95% of bar width
    #             bar.get_y() + bar.get_height()/2,
    #             f'{baselines[i]:}',
    #             ha='right',  # Right-align the text
    #             va='center',
    #             fontsize=10,
    #             color='black')
    
    # plt.yticks(y_pos, baselines)
    plt.gca().yaxis.set_visible(False)

    # plt.grid(axis='x', linestyle='--', alpha=0.2)
    plt.tight_layout(pad=0.2)

    # plt.show()
    plt.savefig(f'{title}-param-ct.png'.lower().replace('/', ' ').replace(' ', '-'))


if __name__ == "__main__":
    # get_from_wandb()
    # exit(0)
    # plot_from_saved()

    """
    param_counts = {
        'CASH': 42469,
        'RNN-EXP': 102149,
        'RNN-IMP': 102149
    }
    plot_parameter_counts(param_counts, 'QMIX / Firefighting')

    param_counts = {
        'CASH': 43365,
        'RNN-EXP': 103173,
        'RNN-IMP': 103173
    }
    plot_parameter_counts(param_counts, 'QMIX / Transport')

    param_counts = {
        'CASH': 26821,
        'RNN-EXP': 118661,
        'RNN-IMP': 118661
    }
    plot_parameter_counts(param_counts, 'MAPPO / Firefighting')

    param_counts = {
        'CASH': 27269,
        'RNN-EXP': 119685,
        'RNN-IMP': 119685
    }
    plot_parameter_counts(param_counts, 'MAPPO / Transport')

    param_counts = {
        'CASH': 39959557,
        'RNN-EXP': 100786181,
        'RNN-IMP': 100786181
    }
    plot_parameter_counts(param_counts, 'DAgger / Firefighting')

    param_counts = {
        'CASH': 39988229,
        'RNN-EXP': 100818949,
        'RNN-IMP': 100818949
    }
    plot_parameter_counts(param_counts, 'DAgger / Transport')
    """

    # for last min LN ablation
    # dagger fire
    filename = "final-dagger-fire-ln.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    plot_metrics(df, y_label='Training Returns', y_column='policy/returns', title='DAgger / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")

    # dagger hmt
    filename = "final-dagger-hmt-ln.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    # HMT
    plot_metrics(df, y_label='Training Returns', y_column='policy/returns', title='DAgger / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")

    # qmix fire
    filename = "ln-ablation-qmix-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    plot_metrics(df, y_label='Training Returns', y_column='returns', title='QMIX / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")

    # QMIX transport
    filename = "ln-ablation-qmix-transport.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    plot_metrics(df, y_label='Training Returns', y_column='returns', title='QMIX / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")

    # mappo fire
    filename = "ln-ablation-mappo-fire.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    plot_metrics(df, y_label='Training Returns', y_column='returns', title='MAPPO / Firefighting', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")

    # mappo transport
    filename = "ln-ablation-mappo-transport.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    def ln_name(row):
        if row['use_ln']: 
            return "CASH"
        else:
            return "CASH w/o LN"

    df['baseline'] = df.apply(lambda row: f"{ln_name(row)}", axis=1)

    plot_metrics(df, y_label='Training Returns', y_column='returns', title='MAPPO / Transport', mean_window=100, std_window=100, downsample_factor=1, save_folder=".")
