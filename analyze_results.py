import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def extract_results(results_dir):
    """Extract metrics from evaluation result directories"""
    results = []
    
    # Traverse all evaluation result directories
    for model_dir in Path(results_dir).glob("*"):
        if not model_dir.is_dir():
            continue
        
        result_file = model_dir / "evaluation_results.txt"
        if not result_file.exists():
            continue
        
        # Parse directory name to get model information
        dir_name = model_dir.name
        
        # Extract quantization type and parameters
        if "rvq_nq" in dir_name:
            parts = dir_name.split("_")
            quantizer_type = "RVQ"
            n_q_max = int(parts[1].replace("nq", ""))
            bins = int(parts[2].replace("bins", ""))
            bitrate = float(parts[3].replace("kbps", ""))
            fsq_levels = None
        elif dir_name.startswith("fsq_"):
            quantizer_type = "FSQ"
            n_q_max = None
            bins = None
            fsq_levels = dir_name.split("_")[1]
            bitrate = None  # Will be extracted from result file
        else:
            continue
        
        # Read result file
        metrics = {}
        model_date = None
        with open(result_file, "r") as f:
            for line in f:
                if "模型:" in line:
                    model_date = line.split(":")[1].strip()
                elif "比特率:" in line:
                    bitrate = float(re.search(r"([\d.]+) kbps", line).group(1))
                elif "PESQ_NB:" in line:
                    metrics["pesq_nb"] = float(re.search(r"([\d.]+)", line).group(1))
                elif "PESQ_WB:" in line:
                    metrics["pesq_wb"] = float(re.search(r"([\d.]+)", line).group(1))
                elif "STOI:" in line:
                    metrics["stoi"] = float(re.search(r"([\d.]+)", line).group(1))
        
        # Add to results list
        results.append({
            "model": dir_name,
            "model_date": model_date,
            "quantizer_type": quantizer_type,
            "n_q_max": n_q_max,
            "bins": bins,
            "fsq_levels": fsq_levels,
            "bitrate": bitrate,
            **metrics
        })
    
    return pd.DataFrame(results)

def plot_rvq_n_q_comparison(df, output_dir):
    """Compare performance of different RVQ codebook layers"""
    # Filter data
    rvq_df = df[(df["quantizer_type"] == "RVQ") & (df["bins"] == 1024)]
    
    if len(rvq_df) < 3:  # Ensure at least 3 different n_q_max data points
        print("Not enough RVQ data for codebook layers comparison")
        return
    
    # Group by bitrate
    bitrates = sorted(rvq_df["bitrate"].unique())
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["pesq_nb", "pesq_wb", "stoi"]
    titles = ["PESQ-NB vs. Codebook Layers", "PESQ-WB vs. Codebook Layers", "STOI vs. Codebook Layers"]
    ylabels = ["PESQ-NB", "PESQ-WB", "STOI"]
    
    # Plot a line for each bitrate
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        for bw in bitrates:
            data = rvq_df[rvq_df["bitrate"] == bw].sort_values("n_q_max")
            ax.plot(data["n_q_max"], data[metric], marker="o", label=f"{bw} kbps", linewidth=2)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Codebook Layers (n_q_max)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([4, 8, 12])  # Explicitly set x-axis ticks
        
        # Add value labels
        for bw in bitrates:
            data = rvq_df[rvq_df["bitrate"] == bw].sort_values("n_q_max")
            for x, y in zip(data["n_q_max"], data[metric]):
                ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                           xytext=(0, 7), ha='center', fontsize=9)
        
        if i == 0:
            ax.legend(fontsize=10)
        
        # Set y-axis range for better visualization
        if metric == "pesq_nb":
            ax.set_ylim(2.0, 2.3)
        elif metric == "pesq_wb":
            ax.set_ylim(1.6, 1.9)
        elif metric == "stoi":
            ax.set_ylim(0.86, 0.90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rvq_layers_comparison.png"), dpi=300)
    plt.close()

def plot_rvq_bins_comparison(df, output_dir):
    """Compare performance of different RVQ codebook sizes"""
    # Filter data
    rvq_df = df[(df["quantizer_type"] == "RVQ") & (df["n_q_max"] == 8)]
    
    if len(rvq_df) < 3:  # Ensure at least 3 different bins data points
        print("Not enough RVQ data for codebook size comparison")
        return
    
    # Group by bitrate
    bitrates = sorted(rvq_df["bitrate"].unique())
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["pesq_nb", "pesq_wb", "stoi"]
    titles = ["PESQ-NB vs. Codebook Size", "PESQ-WB vs. Codebook Size", "STOI vs. Codebook Size"]
    ylabels = ["PESQ-NB", "PESQ-WB", "STOI"]
    
    # Plot a line for each bitrate
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        for bw in bitrates:
            data = rvq_df[rvq_df["bitrate"] == bw].sort_values("bins")
            ax.plot(data["bins"], data[metric], marker="o", label=f"{bw} kbps", linewidth=2)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Codebook Size (bins)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048])  # Explicitly set x-axis ticks
        
        # Add value labels
        for bw in bitrates:
            data = rvq_df[rvq_df["bitrate"] == bw].sort_values("bins")
            for x, y in zip(data["bins"], data[metric]):
                ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                           xytext=(0, 7), ha='center', fontsize=9)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rvq_bins_comparison.png"), dpi=300)
    plt.close()

def plot_fsq_vs_rvq(df, output_dir):
    """Compare performance of FSQ and RVQ"""
    # Filter data - select FSQ and best RVQ configuration
    fsq = df[df["quantizer_type"] == "FSQ"]
    
    # Select the best performing RVQ configuration (n_q_max=12, bins=1024, 1.5kbps)
    best_rvq = df[(df["quantizer_type"] == "RVQ") & 
                  (df["n_q_max"] == 12) & 
                  (df["bins"] == 1024) &
                  (abs(df["bitrate"] - 1.5) < 0.1)]  # Close to 1.5 kbps
    
    if len(best_rvq) == 0 or len(fsq) == 0:
        print("Not enough data for FSQ vs RVQ comparison")
        return
    
    # Merge data
    compare_df = pd.concat([best_rvq, fsq])
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["pesq_nb", "pesq_wb", "stoi"]
    titles = ["PESQ-NB Comparison", "PESQ-WB Comparison", "STOI Comparison"]
    
    # Plot comparison
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        bars = ax.bar(compare_df["model"], compare_df[metric], 
                     color=['#1f77b4' if t == 'RVQ' else '#ff7f0e' for t in compare_df["quantizer_type"]])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                    ha='center', va='bottom', fontsize=10)
        
        # Add bitrate labels
        for j, row in enumerate(compare_df.itertuples()):
            ax.text(j, 0.05, f'{row.bitrate:.2f} kbps', 
                   ha='center', va='bottom', fontsize=9, color='red')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, max(compare_df[metric].max() * 1.1, 1.0))  # Set y-axis limit
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Add quantization type labels
        for j, row in enumerate(compare_df.itertuples()):
            ax.text(j, compare_df[metric].max() * 0.9, row.quantizer_type, 
                   ha='center', fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fsq_vs_rvq.png"), dpi=300)
    plt.close()

def plot_bitrate_vs_quality(df, output_dir):
    """Plot bitrate vs quality relationship"""
    # Color mapping for different configurations
    config_colors = {
        "nq4_bins1024": "#1f77b4",
        "nq8_bins512": "#ff7f0e", 
        "nq8_bins1024": "#2ca02c",
        "nq8_bins2048": "#d62728",
        "nq12_bins1024": "#9467bd",
        "fsq_8655": "#8c564b"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["pesq_nb", "pesq_wb", "stoi"]
    titles = ["PESQ-NB vs. Bitrate", "PESQ-WB vs. Bitrate", "STOI vs. Bitrate"]
    
    # Group data for plotting
    grouped_data = []
    
    # RVQ configurations
    for n_q in df["n_q_max"].dropna().unique():
        for bins in df["bins"].dropna().unique():
            subset = df[(df["quantizer_type"] == "RVQ") & 
                        (df["n_q_max"] == n_q) & 
                        (df["bins"] == bins)]
            
            if len(subset) >= 2:  # Need at least 2 points to draw a line
                config_key = f"nq{int(n_q)}_bins{int(bins)}"
                grouped_data.append({
                    "data": subset.sort_values("bitrate"),
                    "label": f"RVQ nq={int(n_q)} bins={int(bins)}",
                    "color": config_colors.get(config_key, "#333333"),
                    "marker": "o"
                })
    
    # FSQ configurations
    for levels in df["fsq_levels"].dropna().unique():
        subset = df[df["fsq_levels"] == levels]
        if not subset.empty:
            config_key = f"fsq_{levels}"
            grouped_data.append({
                "data": subset,
                "label": f"FSQ {levels}",
                "color": config_colors.get(config_key, "#333333"),
                "marker": "^"
            })
    
    # Plot charts
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        for group in grouped_data:
            data = group["data"]
            ax.plot(data["bitrate"], data[metric], 
                   marker=group["marker"], label=group["label"],
                   color=group["color"], linewidth=2, markersize=8)
            
            # Add value labels to each point
            for x, y in zip(data["bitrate"], data[metric]):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                           xytext=(0,7), ha='center', fontsize=8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Bitrate (kbps)", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend in the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bitrate_vs_quality.png"), dpi=300)
    plt.close()

def create_radar_chart(df, output_dir):
    """Create radar charts comparing different configurations for each bitrate"""
    # Group by bitrate
    bitrate_groups = {
        "Low Bitrate": (0, 2.0),  # 0-2.0 kbps
        "Medium Bitrate": (2.0, 4.0),  # 2.0-4.0 kbps
        "High Bitrate": (4.0, 7.0)   # 4.0-7.0 kbps
    }
    
    metrics = ["pesq_nb", "pesq_wb", "stoi"]
    # Scale STOI to make it more comparable with PESQ on radar chart
    scale_stoi = lambda x: x * 5  # Scale 0-1 range to 0-5
    
    for group_name, (min_bw, max_bw) in bitrate_groups.items():
        bw_data = df[(df["bitrate"] >= min_bw) & (df["bitrate"] < max_bw)]
        
        if len(bw_data) < 2:
            continue
            
        # Create identifier for each model configuration
        bw_data["config_id"] = bw_data.apply(
            lambda row: f"{row['quantizer_type']}-{row['n_q_max'] if row['n_q_max'] else 'FSQ'}-{row['bins'] if row['bins'] else row['fsq_levels']}", 
            axis=1
        )
        
        # Prepare radar chart data
        configs = bw_data["config_id"].unique()
        num_configs = len(configs)
        
        if num_configs > 0:
            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set angles
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the radar chart
            
            # Draw a line for each configuration
            for config in configs:
                config_data = bw_data[bw_data["config_id"] == config]
                if len(config_data) == 0:
                    continue
                
                values = [
                    config_data["pesq_nb"].values[0],
                    config_data["pesq_wb"].values[0],
                    scale_stoi(config_data["stoi"].values[0])
                ]
                values += values[:1]  # Close the radar chart
                
                ax.plot(angles, values, linewidth=2, label=f"{config} ({config_data['bitrate'].values[0]:.1f}kbps)")
                ax.fill(angles, values, alpha=0.1)
            
            # Set radar chart labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(["PESQ-NB", "PESQ-WB", "STOI×5"])
            
            # Add legend and title
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title(f"Performance Comparison at {group_name}", fontsize=15, pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"radar_chart_{group_name.replace(' ', '_')}.png"), dpi=300)
            plt.close()

def create_summary_table(df, output_dir):
    """Create results summary table"""
    # Reorganize data for readability
    summary = df.copy()
    
    # Format FSQ level information
    summary["config"] = summary.apply(
        lambda row: f"nq={row['n_q_max']}, bins={row['bins']}" if row["quantizer_type"] == "RVQ" 
                    else f"levels={row['fsq_levels']}", axis=1
    )
    
    # Add model size estimate column
    summary["model_size_estimate"] = summary.apply(
        lambda row: "Large" if row["n_q_max"] == 12 else 
                   "Medium" if row["n_q_max"] == 8 else 
                   "Small" if row["n_q_max"] == 4 else 
                   "Very Small" if row["quantizer_type"] == "FSQ" else "Unknown", 
        axis=1
    )
    
    # Add composite score column (PESQ-NB * 0.4 + PESQ-WB * 0.3 + STOI * 0.3)
    summary["composite_score"] = summary.apply(
        lambda row: row["pesq_nb"] * 0.4 + row["pesq_wb"] * 0.3 + row["stoi"] * 0.3,
        axis=1
    )
    
    # Select columns to display
    display_cols = ["model", "quantizer_type", "config", "model_size_estimate", 
                   "bitrate", "pesq_nb", "pesq_wb", "stoi", "composite_score"]
    summary = summary[display_cols].sort_values(["quantizer_type", "bitrate"])
    
    # Save as CSV
    summary.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    # Group by bitrate and find best performing model for each metric
    best_models = []
    for metric in ["pesq_nb", "pesq_wb", "stoi", "composite_score"]:
        # Group by approximate bitrate
        bitrate_groups = {
            "Low (~0.5 kbps)": (0, 1.0),
            "Medium (~1.5 kbps)": (1.0, 2.0),
            "High (~3.0 kbps)": (2.0, 4.0),
            "Very High (~6.0 kbps)": (4.0, 8.0)
        }
        
        for group_name, (min_bw, max_bw) in bitrate_groups.items():
            bw_data = summary[(summary["bitrate"] >= min_bw) & (summary["bitrate"] < max_bw)]
            if not bw_data.empty:
                best = bw_data.loc[bw_data[metric].idxmax()]
                best_models.append({
                    "bitrate_group": group_name,
                    "actual_bitrate": best["bitrate"],
                    "metric": metric,
                    "best_model": best["model"],
                    "quantizer_type": best["quantizer_type"],
                    "config": best["config"],
                    "value": best[metric]
                })
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv(os.path.join(output_dir, "best_models_by_metric.csv"), index=False)
    
    return summary, best_df

def main():
    results_dir = "/home/chenkuangwei/chenkuangwei_nfs_data/rvqgan/codec/evaluation_results"
    output_dir = "/home/chenkuangwei/chenkuangwei_nfs_data/rvqgan/codec/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    df = extract_results(results_dir)
    
    if df.empty:
        print("No evaluation result data found")
        return
    
    # Create results table
    summary, best_models = create_summary_table(df, output_dir)
    print("Results Summary:")
    print(summary)
    
    print("\nBest Models (by metric):")
    print(best_models)
    
    # Plot comparison charts
    plot_rvq_n_q_comparison(df, output_dir)
    plot_rvq_bins_comparison(df, output_dir)
    plot_fsq_vs_rvq(df, output_dir)
    plot_bitrate_vs_quality(df, output_dir)
    create_radar_chart(df, output_dir)
    
    print(f"Analysis results saved to {output_dir} directory")

if __name__ == "__main__":
    main()