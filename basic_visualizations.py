import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and display available cleaned datasets"""
    cleaned_files = glob.glob("cleaned/*.csv")
    
    if not cleaned_files:
        print("No cleaned CSV files found in the cleaned directory.")
        return None
    
    print("Available cleaned CSV files:")
    for i, file in enumerate(cleaned_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    try:
        choice = int(input(f"\nEnter the number of the file you want to visualize (1-{len(cleaned_files)}): "))
        if 1 <= choice <= len(cleaned_files):
            file_path = cleaned_files[choice-1]
            print(f"Loading: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df, os.path.basename(file_path)
        else:
            print("Invalid choice.")
            return None, None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None, None

def create_scatter_plots(df, numerical_cols, filename_prefix):
    """Create scatter plots for numerical columns"""
    if len(numerical_cols) < 2:
        print("Need at least 2 numerical columns for scatter plots.")
        return
    
    print("\nCreating scatter plots...")
    
    # Create pairwise scatter plots for first few numerical columns
    cols_to_plot = numerical_cols[:min(4, len(numerical_cols))]  # Limit to 4 columns for readability
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(cols_to_plot)-1, len(cols_to_plot)-1, figsize=(12, 10))
    fig.suptitle(f'Scatter Plot Matrix - {filename_prefix}', fontsize=16)
    
    # Handle case where we only have 2 columns (single subplot)
    if len(cols_to_plot) == 2:
        axes = [axes]
    elif len(cols_to_plot) == 3:
        axes = axes.flatten()
    
    plot_count = 0
    for i in range(len(cols_to_plot)-1):
        for j in range(i+1, len(cols_to_plot)):
            if len(cols_to_plot) == 2:
                ax = axes[plot_count] if len(cols_to_plot) > 2 else axes
            elif len(cols_to_plot) == 3:
                ax = axes[plot_count]
            else:
                ax = axes[i, j-i-1]
                
            col1, col2 = cols_to_plot[i], cols_to_plot[j]
            ax.scatter(df[col1], df[col2], alpha=0.6)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            
            # Calculate and display correlation
            corr = df[[col1, col2]].corr().iloc[0, 1]
            ax.set_title(f'{col1} vs {col2}\nCorrelation: {corr:.2f}')
            
            plot_count += 1
    
    # Hide unused subplots
    if len(cols_to_plot) > 2:
        for i in range(plot_count, len(axes)):
            axes.flatten()[i].set_visible(False)
    
    plt.tight_layout()
    scatter_filename = f"{filename_prefix}_scatter_plots.png"
    plt.savefig(scatter_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Scatter plots saved as '{scatter_filename}'")

def create_bar_plots(df, categorical_cols, numerical_cols, filename_prefix):
    """Create bar plots for categorical data"""
    print("\nCreating bar plots...")
    
    # For categorical columns, create value count bar plots
    if len(categorical_cols) > 0:
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            bars = plt.bar(range(len(value_counts)), value_counts.values, color='skyblue')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.title(f'Distribution of {col}')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            bar_filename = f"{filename_prefix}_{col}_bar_plot.png"
            plt.savefig(bar_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Bar plot for '{col}' saved as '{bar_filename}'")
    
    # For numerical columns, create bar plots of means
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 6))
        means = [df[col].mean() for col in numerical_cols[:10]]  # Limit to first 10 columns
        bars = plt.bar(range(len(means)), means, color='lightcoral')
        plt.xlabel('Features')
        plt.ylabel('Mean Value')
        plt.title(f'Mean Values of Numerical Features - {filename_prefix}')
        plt.xticks(range(len(means)), numerical_cols[:10], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        bar_filename = f"{filename_prefix}_numerical_means_bar_plot.png"
        plt.savefig(bar_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Numerical means bar plot saved as '{bar_filename}'")

def create_line_charts(df, numerical_cols, filename_prefix):
    """Create line charts for numerical data"""
    print("\nCreating line charts...")
    
    if len(numerical_cols) > 0:
        # If we have a sequential index or time column, we can create meaningful line charts
        # For now, we'll create a line chart of sorted values for each numerical column
        for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
            plt.figure(figsize=(12, 6))
            # Sort values to show distribution pattern
            sorted_values = df[col].sort_values().reset_index(drop=True)
            plt.plot(sorted_values.index, sorted_values.values, marker='o', linewidth=2, markersize=3)
            plt.xlabel('Index (sorted)')
            plt.ylabel(col)
            plt.title(f'Distribution Pattern of {col} (Sorted Values)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            line_filename = f"{filename_prefix}_{col}_line_chart.png"
            plt.savefig(line_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Line chart for '{col}' saved as '{line_filename}'")

def main():
    """Main function to run the basic visualization task"""
    print("=" * 60)
    print("BASIC DATA VISUALIZATION TOOL - TASK 3")
    print("=" * 60)
    print("This tool will create:")
    print("- Scatter plots to visualize relationships")
    print("- Bar plots to show distributions")
    print("- Line charts to display trends")
    print()
    
    # Load data
    df, filename = load_data()
    
    if df is None:
        return
    
    # Extract file name without extension for naming outputs
    filename_prefix = os.path.splitext(filename)[0].replace(' ', '_').replace(')', '').replace('(', '')
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Numerical Columns: {len(numerical_cols)}")
    print(f"  Categorical Columns: {len(categorical_cols)}")
    
    # Create visualizations
    if len(numerical_cols) > 0:
        create_scatter_plots(df, numerical_cols, filename_prefix)
        create_line_charts(df, numerical_cols, filename_prefix)
    
    if len(categorical_cols) > 0 or len(numerical_cols) > 0:
        create_bar_plots(df, categorical_cols, numerical_cols, filename_prefix)
    
    print(f"\n{'='*60}")
    print("BASIC VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print("All visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main()
