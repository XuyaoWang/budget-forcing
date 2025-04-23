import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any
from tqdm import tqdm

def calculate_distinct_n(text: str, tokenizer: Any, n_values: List[int] = [1, 2, 3, 4]) -> Dict[str, float]:
    """
    Calculate Distinct-n metrics for a given text.
    
    Args:
        text: Input text string
        tokenizer: Tokenizer object with an encode method
        n_values: List of n values for which to calculate Distinct-n
        
    Returns:
        Dictionary containing Distinct-n metrics for different n values
    """
    # If tokenizer is not provided, return zeros
    if tokenizer is None:
        return {f'distinct-{n}': 0.0 for n in n_values}
    
    # Tokenize the text
    tokens = tokenizer(text)["input_ids"]
    result = {}
    
    for n in n_values:
        if len(tokens) < n:
            result[f'distinct-{n}'] = 0.0
            continue
            
        # Generate n-grams from tokens
        n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Calculate ratio of unique n-grams to total n-grams
        unique_count = len(set(n_grams))
        total_count = len(n_grams)
        result[f'distinct-{n}'] = unique_count / total_count if total_count > 0 else 0.0
        
    return result

def plot_accuracy_and_length(
    model_name: str,
    results: List[Dict[str, Any]],
    save_dir: str
) -> None:
    """
    Plot accuracy and response length vs. budget forcing times
    
    Args:
        model_name: Name of the model
        results: List of result dictionaries containing num_ignore, accuracy, token_mean, token_std
        save_dir: Directory to save the generated plots
    """
    # Set figure size and style
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14
    })
    
    # Create main axes for accuracy
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # Create second y-axis for response length
    
    # Sort results by num_ignore
    results.sort(key=lambda x: x["num_ignore"])
    
    # Extract data
    num_ignore = [item["num_ignore"] for item in results]
    accuracy = [item["accuracy"] * 100 for item in results]  # Convert to percentage
    token_mean = [item["token_mean"] / 1000 for item in results]  # Convert to thousands
    token_std = [item["token_std"] / 1000 for item in results]  # Standard deviation in thousands
    
    # Plot accuracy on left y-axis (indigo color)
    acc_line, = ax1.plot(num_ignore, accuracy, '-', color='indigo', marker='o', 
                       markersize=6, markerfacecolor='indigo', linewidth=2, 
                       label="Accuracy")
    
    # Plot response length on right y-axis with shaded standard deviation (orange color)
    len_line, = ax2.plot(num_ignore, token_mean, '--', color='darkorange', marker='s', 
                       markersize=6, markerfacecolor='darkorange', linewidth=2, 
                       label="Response Length")
    
    # Add shaded area to represent variance/standard deviation
    ax2.fill_between(num_ignore, 
                     [m - s for m, s in zip(token_mean, token_std)],
                     [m + s for m, s in zip(token_mean, token_std)],
                     color='darkorange', alpha=0.2)
    
    # Set labels and title
    ax1.set_xlabel('Budget Forcing Times', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Accuracy (%)', color='indigo', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Response Length (K)', color='darkorange', fontweight='bold', fontsize=16)
    plt.title(f'{model_name} Performance vs Budget Forcing Times', fontweight='bold', fontsize=18)
    
    # Set tick colors
    ax1.tick_params(axis='y', labelcolor='indigo')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.3, color='lightgray')
    
    # Create combined legend
    plt.legend([acc_line, len_line], ['Accuracy', 'Response Length'], 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=True, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    
    # Save figure if path provided
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{model_name}_perf_vs_budget.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_perf_vs_budget.pdf", bbox_inches='tight')
    plt.close()

def plot_detailed_metrics(
    model_name: str,
    results: List[Dict[str, Any]],
    save_dir: str,
    distinct_metrics: bool = True
) -> None:
    """
    Plot detailed metrics in a 2x2 grid
    
    Args:
        model_name: Name of the model
        results: List of result dictionaries containing metrics
        save_dir: Directory to save the generated plots
        distinct_metrics: Whether to include distinct-n metrics (if available)
    """
    # Sort results by num_ignore
    results.sort(key=lambda x: x["num_ignore"])
    
    # Extract plotting data
    num_ignores = [r["num_ignore"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    token_means = [r["token_mean"] for r in results]
    token_stds = [r["token_std"] for r in results]
    
    # Check if distinct metrics are available - always plot them if they exist
    has_distinct = all(["distinct-1" in r for r in results])
    
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    
    # Get default font size and calculate 1.3x size for titles
    default_fontsize = plt.rcParams['font.size']
    title_fontsize = default_fontsize * 1.3
    
    # Subplot 1: Accuracy
    axs[0, 0].plot(num_ignores, accuracies, marker='o', linestyle='-', color='tab:red', label='Accuracy')
    axs[0, 0].set_title('Accuracy vs. Number of Ignored Tokens', fontsize=title_fontsize)
    axs[0, 0].set_xlabel('Num Ignore')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend(loc='best')
    axs[0, 0].grid(True)
    
    # Subplot 2: Average Token Length
    axs[0, 1].plot(num_ignores, token_means, marker='s', linestyle='-', color='tab:blue', label='Average Token Length')
    axs[0, 1].set_title('Average Token Length vs. Number of Ignored Tokens', fontsize=title_fontsize)
    axs[0, 1].set_xlabel('Num Ignore')
    axs[0, 1].set_ylabel('Token Mean')
    axs[0, 1].legend(loc='best')
    axs[0, 1].grid(True)
    
    # Subplot 3: Token Length Standard Deviation
    axs[1, 0].plot(num_ignores, token_stds, marker='^', linestyle='-', color='tab:green', label='Token Length Standard Deviation')
    axs[1, 0].set_title('Token Length Standard Deviation vs. Number of Ignored Tokens', fontsize=title_fontsize)
    axs[1, 0].set_xlabel('Num Ignore')
    axs[1, 0].set_ylabel('Token Std')
    axs[1, 0].legend(loc='best')
    axs[1, 0].grid(True)
    
    # Subplot 4: Distinct-n metrics (if available)
    if has_distinct:
        # Extract distinct metrics
        distinct_1 = [r.get("distinct-1", 0) for r in results]
        distinct_2 = [r.get("distinct-2", 0) for r in results]
        distinct_3 = [r.get("distinct-3", 0) for r in results]
        distinct_4 = [r.get("distinct-4", 0) for r in results]
        
        colors = ['#CC0000', '#00CC00', '#0000CC', '#CCCC00']  # Red, Green, Blue, Yellow
        markers = ['o', 's', '^', 'd']
        distinct_data = [distinct_1, distinct_2, distinct_3, distinct_4]
        distinct_labels = ['Distinct-1', 'Distinct-2', 'Distinct-3', 'Distinct-4']
        
        for i, (data, label, color, marker) in enumerate(zip(distinct_data, distinct_labels, colors, markers)):
            axs[1, 1].plot(num_ignores, data, marker=marker, linestyle='-', color=color, label=label)
        
        axs[1, 1].set_title('Distinct-n Metrics vs. Number of Ignored Tokens', fontsize=title_fontsize)
        axs[1, 1].set_xlabel('Num Ignore')
        axs[1, 1].set_ylabel('Distinct-n Value')
        axs[1, 1].legend(loc='best')
        axs[1, 1].grid(True)
    else:
        # If distinct metrics not available, plot accuracy vs token mean
        axs[1, 1].plot(token_means, accuracies, marker='*', linestyle='-', color='purple', label='Accuracy vs Token Length')
        axs[1, 1].set_title('Accuracy vs. Token Length', fontsize=title_fontsize)
        axs[1, 1].set_xlabel('Token Mean')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend(loc='best')
        axs[1, 1].grid(True)
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{model_name}_detailed_metrics.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_detailed_metrics.pdf", bbox_inches='tight')
    plt.close()

def format_results_for_plotting(evaluation_results: Dict[str, Any], tokenizer=None) -> List[Dict[str, Any]]:
    """
    Format evaluation results for plotting
    
    Args:
        evaluation_results: Dictionary of evaluation results
        tokenizer: Optional tokenizer for calculating token statistics
        
    Returns:
        List of formatted results for plotting
    """
    formatted_results = []
    
    for ignore_idx, eval_data in evaluation_results.items():
        token_lengths = []
        distinct_values = {"distinct-1": [], "distinct-2": [], "distinct-3": [], "distinct-4": []}
        
        # Extract token lengths and calculate distinct-n metrics from detailed results if tokenizer is provided
        if "detailed_results" in eval_data:
            for item in tqdm(eval_data["detailed_results"], desc=f"Processing details for ignore_idx={ignore_idx}", leave=False):
                if "response" in item:
                    response_text = item["response"]
                    
                    # Calculate token length
                    if tokenizer:
                        tokens = tokenizer(response_text)["input_ids"]
                        token_lengths.append(len(tokens))
                    
                    # Calculate distinct-n metrics if tokenizer is provided
                    if tokenizer:
                        distinct_metrics = calculate_distinct_n(response_text, tokenizer)
                        for n in [1, 2, 3, 4]:
                            key = f"distinct-{n}"
                            distinct_values[key].append(distinct_metrics[key])
        
        # Calculate token statistics
        token_mean = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        token_std = np.std(token_lengths) if token_lengths else 0
        
        # Calculate average distinct-n values
        distinct_means = {}
        for n in [1, 2, 3, 4]:
            key = f"distinct-{n}"
            distinct_means[key] = sum(distinct_values[key]) / len(distinct_values[key]) if distinct_values[key] else 0.0
        
        # Format result entry
        result_entry = {
            "num_ignore": int(ignore_idx),
            "accuracy": eval_data.get("accuracy", 0),
            "token_mean": token_mean,
            "token_std": token_std
        }
        
        # Add distinct-n metrics if calculated
        if any(distinct_values.values()):
            for n in [1, 2, 3, 4]:
                result_entry[f"distinct-{n}"] = distinct_means[f"distinct-{n}"]
        
        formatted_results.append(result_entry)
    
    return formatted_results 