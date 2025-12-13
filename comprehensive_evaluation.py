"""
Comprehensive evaluation plots for ML Lab3 requirements:
1. Evaluating the effectiveness of continuous prediction
2. Verifying the ability to preserve ranks
3. Evaluating the ability to learn from nonlinear data
4. Evaluating stability with noise and large datasets
5. Comparing with traditional regression methods
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats


def load_results_with_noise(result_base_dir):
    """Load results from noise level folders (10, 30, 50)"""
    noise_levels = ['10', '30', '50']
    datasets = ['bike', 'protein', 'sine']
    
    all_data = []
    
    for noise in noise_levels:
        noise_dir = os.path.join(result_base_dir, noise)
        if not os.path.exists(noise_dir):
            continue
            
        for dataset in datasets:
            dataset_dir = os.path.join(noise_dir, dataset)
            search_pattern = os.path.join(dataset_dir, 'results_*.csv')
            files = glob.glob(search_pattern)
            
            for f in files:
                filename = os.path.basename(f)
                parts = filename.replace('.csv', '').split('_')
                
                # Parse loss function name
                if 'sharing' in parts:
                    loss_start_idx = parts.index('sharing') + 1
                else:
                    loss_start_idx = 2
                
                loss = "_".join(parts[loss_start_idx:-2])
                
                df = pd.read_csv(f)
                df = df[pd.to_numeric(df['fold'], errors='coerce').notna()]
                
                if not df.empty:
                    df['Loss'] = loss
                    df['Dataset'] = dataset
                    df['Noise_Level'] = int(noise)
                    all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_base_results(result_base_dir):
    """Load base results (no noise)"""
    datasets = ['bike', 'protein', 'sine']
    all_data = []
    
    for dataset in datasets:
        dataset_dir = os.path.join(result_base_dir, dataset, 'results')
        search_pattern = os.path.join(dataset_dir, 'results_*.csv')
        files = glob.glob(search_pattern)
        
        for f in files:
            filename = os.path.basename(f)
            parts = filename.replace('.csv', '').split('_')
            
            if 'sharing' in parts:
                loss_start_idx = parts.index('sharing') + 1
            else:
                loss_start_idx = 2
            
            loss = "_".join(parts[loss_start_idx:-2])
            
            df = pd.read_csv(f)
            df = df[pd.to_numeric(df['fold'], errors='coerce').notna()]
            
            if not df.empty:
                df['Loss'] = loss
                df['Dataset'] = dataset
                df['Noise_Level'] = 0
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def plot_continuous_prediction_effectiveness(df, output_dir):
    """
    Requirement 1: Evaluating the effectiveness of continuous prediction
    Shows MAE and RMSE across all loss functions
    Creates 3 separate figures - one for each dataset
    """
    print("\n1. Generating Continuous Prediction Effectiveness plots...")
    
    # Aggregate data
    metrics_agg = df.groupby(['Dataset', 'Loss']).agg({
        'test_MAE': ['mean', 'std'],
        'test_RMSE': ['mean', 'std']
    }).reset_index()
    
    datasets = df['Dataset'].unique()
    dataset_names = {'bike': 'Bike Sharing', 'protein': 'Protein', 'sine': 'Sine'}
    colors = {'MAE': '#FFA500', 'GAR': '#60BD68', 'ConR': '#5DA5DA', 
              'Huber': '#FAA43A', 'focal-MAE': '#F17CB0', 'GAR-EXP': '#B2912F'}
    
    # Create separate figure for each dataset
    for dataset in datasets:
        dataset_data = metrics_agg[metrics_agg['Dataset'] == dataset]
        
        if dataset_data.empty:
            continue
        
        # Create figure with 2 subplots (MAE and RMSE)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Continuous Prediction: {dataset_names.get(dataset, dataset)}', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Get loss functions for this dataset
        losses = []
        mae_means = []
        mae_stds = []
        rmse_means = []
        rmse_stds = []
        
        for _, row in dataset_data.iterrows():
            loss = row[('Loss', '')]
            losses.append(loss)
            mae_means.append(row[('test_MAE', 'mean')])
            mae_stds.append(row[('test_MAE', 'std')])
            rmse_means.append(row[('test_RMSE', 'mean')])
            rmse_stds.append(row[('test_RMSE', 'std')])
        
        x_positions = np.arange(len(losses))
        width = 0.6
        
        # MAE plot
        ax = axes[0]
        bars = ax.bar(x_positions, mae_means, width, yerr=mae_stds,
                     color=[colors.get(loss, '#999999') for loss in losses],
                     alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        
        # Highlight best (lowest MAE)
        best_idx = np.argmin(mae_means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        ax.set_title('MAE Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(losses, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, mae_means, mae_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # RMSE plot
        ax = axes[1]
        bars = ax.bar(x_positions, rmse_means, width, yerr=rmse_stds,
                     color=[colors.get(loss, '#999999') for loss in losses],
                     alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        
        # Highlight best (lowest RMSE)
        best_idx = np.argmin(rmse_means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
        ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
        ax.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(losses, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, rmse_means, rmse_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'1_continuous_prediction_{dataset}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   Saved: 1_continuous_prediction_{dataset}.png")
        plt.close()


def plot_rank_preservation(df, output_dir):
    """
    Requirement 2: Verifying the ability to preserve ranks
    Focus on Spearman correlation (rank correlation)
    """
    print("\n2. Generating Rank Preservation plots...")
    
    # Aggregate Spearman and Pearson correlations
    corr_agg = df.groupby(['Dataset', 'Loss']).agg({
        'test_Spearman': ['mean', 'std'],
        'test_Pearson': ['mean', 'std']
    }).reset_index()
    
    datasets = df['Dataset'].unique()
    dataset_names = {'bike': 'Bike Sharing', 'protein': 'Protein', 'sine': 'Sine'}
    colors = {'MAE': '#FFA500', 'GAR': '#60BD68', 'ConR': '#5DA5DA', 
              'Huber': '#FAA43A', 'focal-MAE': '#F17CB0', 'GAR-EXP': '#B2912F'}
    
    # Create subplot for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Rank Preservation Ability (Spearman Correlation)', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = corr_agg[corr_agg['Dataset'] == dataset].sort_values(
            ('test_Spearman', 'mean'), ascending=False)
        
        losses = dataset_data['Loss'].values
        means = dataset_data[('test_Spearman', 'mean')].values
        stds = dataset_data[('test_Spearman', 'std')].values
        
        bar_colors = [colors.get(loss, '#999999') for loss in losses]
        
        bars = ax.barh(losses, means, xerr=stds, color=bar_colors, 
                      alpha=0.8, capsize=4, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            width = bar.get_width()
            ax.text(width + std + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{mean:.3f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Spearman Correlation', fontsize=11, fontweight='bold')
        ax.set_title(dataset_names.get(dataset, dataset), fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.9 threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_rank_preservation.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 2_rank_preservation.png")
    plt.close()


def plot_nonlinear_learning(df, output_dir):
    """
    Requirement 3: Evaluating the ability to learn from nonlinear data
    Focus on Sine dataset performance
    """
    print("\n3. Generating Nonlinear Learning plots...")
    
    # Filter for sine dataset
    sine_data = df[df['Dataset'] == 'sine']
    
    if sine_data.empty:
        print("   No sine dataset found!")
        return
    
    # Aggregate metrics
    sine_agg = sine_data.groupby('Loss').agg({
        'test_MAE': ['mean', 'std'],
        'test_RMSE': ['mean', 'std'],
        'test_Pearson': ['mean', 'std'],
        'test_Spearman': ['mean', 'std']
    }).reset_index()
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Nonlinear Data Learning: Sine Dataset Performance', 
                 fontsize=16, fontweight='bold')
    
    colors = {'MAE': '#FFA500', 'GAR': '#60BD68', 'ConR': '#5DA5DA', 
              'Huber': '#FAA43A', 'focal-MAE': '#F17CB0', 'GAR-EXP': '#B2912F'}
    
    metrics = [
        ('test_MAE', 'Mean Absolute Error', 0, 0, True),
        ('test_RMSE', 'Root Mean Square Error', 0, 1, True),
        ('test_Pearson', 'Pearson Correlation', 1, 0, False),
        ('test_Spearman', 'Spearman Correlation', 1, 1, False)
    ]
    
    for metric, title, row, col, lower_better in metrics:
        ax = axes[row, col]
        
        # Sort by metric
        sorted_data = sine_agg.sort_values((metric, 'mean'), ascending=lower_better)
        
        losses = sorted_data['Loss'].values
        means = sorted_data[(metric, 'mean')].values
        stds = sorted_data[(metric, 'std')].values
        
        bar_colors = [colors.get(loss, '#999999') for loss in losses]
        
        bars = ax.bar(range(len(losses)), means, yerr=stds, 
                     color=bar_colors, alpha=0.8, capsize=5,
                     edgecolor='black', linewidth=1)
        
        # Highlight best performance
        best_idx = 0
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(range(len(losses)))
        ax.set_xticklabels(losses, rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_nonlinear_learning.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 3_nonlinear_learning.png")
    plt.close()


def plot_noise_stability(df_noise, output_dir):
    """
    Requirement 4: Evaluating stability with noise and large datasets
    """
    print("\n4. Generating Noise Stability plots...")
    
    if df_noise.empty:
        print("   No noise data found!")
        return
    
    # Focus on key metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stability Analysis: Performance under Different Noise Levels', 
                 fontsize=16, fontweight='bold')
    
    colors = {'MAE': '#FFA500', 'GAR': '#60BD68', 'ConR': '#5DA5DA', 
              'Huber': '#FAA43A', 'focal-MAE': '#F17CB0', 'GAR-EXP': '#B2912F'}
    
    metrics = [
        ('test_MAE', 'MAE Performance', 0, 0),
        ('test_RMSE', 'RMSE Performance', 0, 1),
        ('test_Pearson', 'Pearson Correlation', 1, 0),
        ('test_Spearman', 'Spearman Correlation', 1, 1)
    ]
    
    for metric, title, row, col in metrics:
        ax = axes[row, col]
        
        # Aggregate by loss, dataset, and noise level
        agg_data = df_noise.groupby(['Loss', 'Dataset', 'Noise_Level'])[metric].mean().reset_index()
        
        # Plot lines for each loss function
        for loss in agg_data['Loss'].unique():
            loss_data = agg_data[agg_data['Loss'] == loss]
            
            for dataset in loss_data['Dataset'].unique():
                dataset_loss = loss_data[loss_data['Dataset'] == dataset].sort_values('Noise_Level')
                
                if len(dataset_loss) > 0:
                    color = colors.get(loss, '#999999')
                    linestyle = '--' if dataset == 'sine' else ('-' if dataset == 'bike' else ':')
                    
                    ax.plot(dataset_loss['Noise_Level'], dataset_loss[metric],
                           marker='o', label=f'{loss} ({dataset})', 
                           color=color, linestyle=linestyle, linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('test_', ''), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([10, 30, 50])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_noise_stability.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 4_noise_stability.png")
    plt.close()


def plot_traditional_comparison(df, output_dir):
    """
    Requirement 5: Comparing with traditional regression methods
    Traditional: MAE, Huber
    Advanced: GAR, ConR, focal-MAE, GAR-EXP
    """
    print("\n5. Generating Traditional vs Advanced Methods comparison...")
    
    # Categorize methods
    traditional = ['MAE', 'Huber']
    advanced = ['GAR', 'ConR', 'focal-MAE', 'GAR-EXP']
    
    # Aggregate data
    agg_data = df.groupby(['Dataset', 'Loss']).agg({
        'test_MAE': 'mean',
        'test_RMSE': 'mean',
        'test_Pearson': 'mean',
        'test_Spearman': 'mean'
    }).reset_index()
    
    # Create comparison table
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Traditional vs Advanced Regression Methods Comparison', 
                 fontsize=16, fontweight='bold')
    
    datasets = df['Dataset'].unique()
    dataset_names = {'bike': 'Bike Sharing', 'protein': 'Protein', 'sine': 'Sine'}
    
    metrics = [
        ('test_MAE', 'Mean Absolute Error (↓)', 0, 0),
        ('test_RMSE', 'Root Mean Square Error (↓)', 0, 1),
        ('test_Pearson', 'Pearson Correlation (↑)', 1, 0),
        ('test_Spearman', 'Spearman Correlation (↑)', 1, 1)
    ]
    
    for metric, title, row, col in metrics:
        ax = axes[row, col]
        
        x_pos = 0
        width = 0.35
        
        for dataset in datasets:
            dataset_data = agg_data[agg_data['Dataset'] == dataset]
            
            # Traditional methods
            trad_data = dataset_data[dataset_data['Loss'].isin(traditional)]
            trad_values = trad_data[metric].values
            trad_labels = trad_data['Loss'].values
            
            # Advanced methods
            adv_data = dataset_data[dataset_data['Loss'].isin(advanced)]
            adv_values = adv_data[metric].values
            adv_labels = adv_data['Loss'].values
            
            # Plot traditional
            for i, (label, val) in enumerate(zip(trad_labels, trad_values)):
                ax.bar(x_pos + i * width, val, width, 
                      color='#FFB347', alpha=0.7, edgecolor='black',
                      label='Traditional' if dataset == datasets[0] and i == 0 else '')
            
            # Plot advanced
            for i, (label, val) in enumerate(zip(adv_labels, adv_values)):
                ax.bar(x_pos + len(trad_labels) * width + i * width, val, width,
                      color='#77DD77', alpha=0.7, edgecolor='black',
                      label='Advanced' if dataset == datasets[0] and i == 0 else '')
            
            x_pos += (len(trad_labels) + len(adv_labels)) * width + 0.5
        
        ax.set_ylabel(metric.replace('test_', ''), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_traditional_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 5_traditional_comparison.png")
    plt.close()
    
    # Create summary table
    create_comparison_table(agg_data, traditional, advanced, output_dir)


def create_comparison_table(agg_data, traditional, advanced, output_dir):
    """Create a summary table comparing traditional vs advanced methods"""
    
    print("\n   Creating comparison summary table...")
    
    results = []
    
    for dataset in agg_data['Dataset'].unique():
        dataset_data = agg_data[agg_data['Dataset'] == dataset]
        
        trad_data = dataset_data[dataset_data['Loss'].isin(traditional)]
        adv_data = dataset_data[dataset_data['Loss'].isin(advanced)]
        
        # Calculate averages
        for metric in ['test_MAE', 'test_RMSE', 'test_Pearson', 'test_Spearman']:
            trad_mean = trad_data[metric].mean()
            adv_mean = adv_data[metric].mean()
            
            improvement = ((trad_mean - adv_mean) / trad_mean * 100) if 'MAE' in metric or 'RMSE' in metric else ((adv_mean - trad_mean) / trad_mean * 100)
            
            results.append({
                'Dataset': dataset,
                'Metric': metric.replace('test_', ''),
                'Traditional_Avg': trad_mean,
                'Advanced_Avg': adv_mean,
                'Improvement_%': improvement
            })
    
    df_comparison = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, 'traditional_vs_advanced_summary.csv')
    df_comparison.to_csv(csv_file, index=False)
    print(f"   Saved comparison table: traditional_vs_advanced_summary.csv")
    
    # Print summary
    print("\n   " + "="*80)
    print("   TRADITIONAL vs ADVANCED METHODS SUMMARY")
    print("   " + "="*80)
    print(df_comparison.to_string(index=False))
    print("   " + "="*80)


def main():
    result_dir = 'result'
    output_dir = 'evaluation_plots'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION ANALYSIS")
    print("="*80)
    
    # Load base results
    print("\nLoading base results...")
    df_base = load_base_results(result_dir)
    print(f"Loaded {len(df_base)} records from base results")
    
    # Load noise results
    print("\nLoading noise-level results...")
    df_noise = load_results_with_noise(result_dir)
    if not df_noise.empty:
        print(f"Loaded {len(df_noise)} records from noise experiments")
    
    # Generate all plots
    if not df_base.empty:
        plot_continuous_prediction_effectiveness(df_base, output_dir)
        plot_rank_preservation(df_base, output_dir)
        plot_nonlinear_learning(df_base, output_dir)
        plot_traditional_comparison(df_base, output_dir)
    
    if not df_noise.empty:
        plot_noise_stability(df_noise, output_dir)
    
    print("\n" + "="*80)
    print(f"All evaluation plots saved to: {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
