"""
Outputs:
    - Multiple PNG visualization files
    - summary_statistics.txt with key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import seaborn for better aesthetics
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Plots will use matplotlib defaults.")
    print("For better visualizations: pip install seaborn")

# Configuration
INPUT_FILE = 'dataset_enriched.csv'
OUTPUT_DIR = Path('eda_outputs')
SUMMARY_FILE = OUTPUT_DIR / 'summary_statistics.txt'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load the enriched dataset."""
    print("="*70)
    print("Loading enriched dataset...")
    print("="*70)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Tasks: {sorted(df['Task'].unique())}")
    print(f"Participants: {df['person_id'].nunique()}")

    return df


# ============================================================================
# 1. STRATEGY DISTRIBUTION ANALYSIS
# ============================================================================

def plot_strategy_distributions(df):
    """
    Visualize distribution of primary strategies across all tasks.
    """
    print("\n" + "="*70)
    print("1. Analyzing Strategy Distributions")
    print("="*70)

    # Overall strategy distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strategy Distributions by Task', fontsize=16, fontweight='bold')

    for idx, task in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2, idx % 2]
        task_data = df[df['Task'] == task]

        # Get strategy counts
        strategy_counts = task_data['L3_primary_strategy'].value_counts()

        # Plot
        strategy_counts.plot(kind='barh', ax=ax, color=f'C{idx}')
        ax.set_title(f'Task {task} (n={len(task_data)})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Strategy', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        total = len(task_data)
        for i, (strategy, count) in enumerate(strategy_counts.items()):
            pct = 100 * count / total
            ax.text(count, i, f'  {count} ({pct:.1f}%)',
                   va='center', fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig1_strategy_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Strategy comparison across tasks (stacked bar)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 8 strategies overall
    top_strategies = df['L3_primary_strategy'].value_counts().head(8).index

    # Create cross-tabulation
    strategy_by_task = pd.crosstab(df['Task'], df['L3_primary_strategy'])
    strategy_by_task = strategy_by_task[top_strategies]

    # Plot stacked bar chart
    strategy_by_task.plot(kind='bar', stacked=False, ax=ax, width=0.8)
    ax.set_title('Strategy Comparison Across Tasks (Top 8 Strategies)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticklabels(['Task 1\n(No Restrictions)',
                        'Task 2\n(Deletion Only)',
                        'Task 3\n(No Restrictions)',
                        'Task 4\n(Time Pressure)'],
                       rotation=0)
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig2_strategy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_strategy_coverage(df):
    """
    Identify top strategies that cover 80%+ of data.
    """
    print("\n--- Strategy Coverage Analysis ---")

    strategy_counts = df['L3_primary_strategy'].value_counts()
    strategy_pcts = 100 * strategy_counts / len(df)
    cumulative_pcts = strategy_pcts.cumsum()

    print("\nCumulative coverage by strategy:")
    for strategy, cum_pct in cumulative_pcts.items():
        count = strategy_counts[strategy]
        pct = strategy_pcts[strategy]
        print(f"  {strategy:25s}: {count:3d} ({pct:5.2f}%)  [Cumulative: {cum_pct:5.2f}%]")
        if cum_pct >= 80:
            print(f"\n  → Top {cumulative_pcts[cumulative_pcts <= 80].shape[0] + 1} strategies cover {cum_pct:.1f}% of data")
            break

    return strategy_counts


# ============================================================================
# 2. SECURITY VS MEMORY COST ANALYSIS
# ============================================================================

def plot_security_memory_tradeoff(df):
    """
    Visualize the tradeoff between security gain and memory cost.
    """
    print("\n" + "="*70)
    print("2. Analyzing Security vs. Memory Cost Tradeoffs")
    print("="*70)

    # Filter out rows with missing security data
    df_valid = df.dropna(subset=['security_guesses_log10', 'memory_cost_estimate'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Security vs Memory Cost (all tasks)
    ax = axes[0]

    for task in [1, 2, 3, 4]:
        task_data = df_valid[df_valid['Task'] == task]
        ax.scatter(task_data['memory_cost_estimate'],
                  task_data['security_guesses_log10'],
                  alpha=0.6, s=50, label=f'Task {task}')

    ax.set_xlabel('Memory Cost Estimate (0-1, higher = harder)', fontsize=12)
    ax.set_ylabel('Security (log10 guesses)', fontsize=12)
    ax.set_title('Security vs. Memory Cost Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: By primary strategy
    ax = axes[1]

    # Get top 6 strategies
    top_strategies = df['L3_primary_strategy'].value_counts().head(6).index

    for strategy in top_strategies:
        strategy_data = df_valid[df_valid['L3_primary_strategy'] == strategy]
        ax.scatter(strategy_data['memory_cost_estimate'],
                  strategy_data['security_guesses_log10'],
                  alpha=0.6, s=50, label=strategy)

    ax.set_xlabel('Memory Cost Estimate (0-1, higher = harder)', fontsize=12)
    ax.set_ylabel('Security (log10 guesses)', fontsize=12)
    ax.set_title('Security vs. Memory Cost by Strategy', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig3_security_memory_tradeoff.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Additional: Efficiency plot (security per unit memory cost)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate efficiency metric
    df_valid['efficiency'] = df_valid['security_guesses_log10'] / (df_valid['memory_cost_estimate'] + 0.1)

    # Plot by task
    task_labels = {1: 'Task 1\n(No Restrictions)',
                   2: 'Task 2\n(Deletion Only)',
                   3: 'Task 3\n(No Restrictions)',
                   4: 'Task 4\n(Time Pressure)'}

    efficiency_by_task = [df_valid[df_valid['Task'] == t]['efficiency'].dropna()
                          for t in [1, 2, 3, 4]]

    bp = ax.boxplot(efficiency_by_task, labels=[task_labels[t] for t in [1, 2, 3, 4]],
                    patch_artist=True)

    # Color boxes
    colors = ['C0', 'C1', 'C2', 'C3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Security Efficiency (security / memory_cost)', fontsize=12)
    ax.set_title('Password Editing Efficiency by Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig4_efficiency_by_task.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# 3. ADDITIVE VS SUBTRACTIVE BIAS ANALYSIS
# ============================================================================

def plot_operation_bias(df):
    """
    Analyze additive vs. subtractive operations across tasks.
    """
    print("\n" + "="*70)
    print("3. Analyzing Additive vs. Subtractive Bias")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Operation counts by task
    ax = axes[0, 0]

    task_ops = df.groupby('Task')[['L2_additive_ops_count', 'L2_subtractive_ops_count']].mean()

    x = np.arange(len(task_ops))
    width = 0.35

    ax.bar(x - width/2, task_ops['L2_additive_ops_count'], width,
           label='Additive', color='green', alpha=0.7)
    ax.bar(x + width/2, task_ops['L2_subtractive_ops_count'], width,
           label='Subtractive', color='red', alpha=0.7)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Average Operations per Edit', fontsize=12)
    ax.set_title('Additive vs. Subtractive Operations by Task', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3', 'Task 4'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, task in enumerate([1, 2, 3, 4]):
        add_val = task_ops.loc[task, 'L2_additive_ops_count']
        sub_val = task_ops.loc[task, 'L2_subtractive_ops_count']
        ax.text(i - width/2, add_val, f'{add_val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, sub_val, f'{sub_val:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Operation bias distribution
    ax = axes[0, 1]

    bias_by_task = pd.crosstab(df['Task'], df['L2_operation_bias'], normalize='index') * 100

    bias_by_task.plot(kind='bar', stacked=True, ax=ax,
                     color=['green', 'gray', 'red', 'blue'])
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Operation Bias Distribution by Task', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3', 'Task 4'], rotation=0)
    ax.legend(title='Bias Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Character changes by task
    ax = axes[1, 0]

    char_changes = df.groupby('Task')[['L1_chars_added', 'L1_chars_deleted',
                                        'L1_chars_substituted']].mean()

    char_changes.plot(kind='bar', ax=ax, color=['green', 'red', 'orange'], alpha=0.7)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Average Characters per Edit', fontsize=12)
    ax.set_title('Character-Level Operations by Task', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3', 'Task 4'], rotation=0)
    ax.legend(['Added', 'Deleted', 'Substituted'])
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Dominant operation type
    ax = axes[1, 1]

    dom_op_by_task = pd.crosstab(df['Task'], df['L1_dominant_operation'], normalize='index') * 100

    dom_op_by_task.plot(kind='bar', stacked=True, ax=ax,
                        color=['green', 'gray', 'blue', 'red', 'purple'])
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Dominant Operation Type by Task', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3', 'Task 4'], rotation=0)
    ax.legend(title='Operation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig5_operation_bias.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ============================================================================
# 4. CONSTRAINT EFFECT ANALYSIS
# ============================================================================

def analyze_constraint_effects(df):
    """
    Quantify the effects of constraints on behavior.
    Tasks 1→2: Deletion-only constraint
    Tasks 3→4: Time pressure constraint
    """
    print("\n" + "="*70)
    print("4. Analyzing Constraint Effects")
    print("="*70)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- DELETION-ONLY CONSTRAINT (Tasks 1 → 2) ----
    ax = axes[0]

    # Get paired data (same person, same base passwords)
    # For simplicity, compare overall distributions
    task1_data = df[df['Task'] == 1]
    task2_data = df[df['Task'] == 2]

    metrics_comparison = pd.DataFrame({
        'Task 1 (No Restrictions)': [
            task1_data['L2_additive_ops_count'].mean(),
            task1_data['L2_subtractive_ops_count'].mean(),
            task1_data['L1_chars_added'].mean(),
            task1_data['L1_chars_deleted'].mean(),
            (task1_data['L2_operation_bias'] == 'additive').sum() / len(task1_data) * 100,
            (task1_data['L2_operation_bias'] == 'subtractive').sum() / len(task1_data) * 100
        ],
        'Task 2 (Deletion Only)': [
            task2_data['L2_additive_ops_count'].mean(),
            task2_data['L2_subtractive_ops_count'].mean(),
            task2_data['L1_chars_added'].mean(),
            task2_data['L1_chars_deleted'].mean(),
            (task2_data['L2_operation_bias'] == 'additive').sum() / len(task2_data) * 100,
            (task2_data['L2_operation_bias'] == 'subtractive').sum() / len(task2_data) * 100
        ]
    }, index=['Additive Ops', 'Subtractive Ops', 'Chars Added', 'Chars Deleted',
              '% Additive Bias', '% Subtractive Bias'])

    x = np.arange(len(metrics_comparison))
    width = 0.35

    ax.barh(x - width/2, metrics_comparison['Task 1 (No Restrictions)'], width,
            label='Task 1 (No Restrictions)', color='C0', alpha=0.7)
    ax.barh(x + width/2, metrics_comparison['Task 2 (Deletion Only)'], width,
            label='Task 2 (Deletion Only)', color='C1', alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(metrics_comparison.index)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Deletion-Only Constraint Effect (Tasks 1 → 2)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, metric in enumerate(metrics_comparison.index):
        val1 = metrics_comparison.loc[metric, 'Task 1 (No Restrictions)']
        val2 = metrics_comparison.loc[metric, 'Task 2 (Deletion Only)']
        ax.text(val1, i - width/2, f' {val1:.2f}', va='center', fontsize=9)
        ax.text(val2, i + width/2, f' {val2:.2f}', va='center', fontsize=9)

    # ---- TIME PRESSURE CONSTRAINT (Tasks 3 → 4) ----
    ax = axes[1]

    task3_data = df[df['Task'] == 3]
    task4_data = df[df['Task'] == 4]

    metrics_comparison2 = pd.DataFrame({
        'Task 3 (No Restrictions)': [
            task3_data['L2_additive_ops_count'].mean(),
            task3_data['L2_subtractive_ops_count'].mean(),
            task3_data['L3_is_rewrite'].sum() / len(task3_data) * 100,
            task3_data['L3_is_minimal'].sum() / len(task3_data) * 100,
            task3_data['L1_edit_distance'].mean(),
            task3_data['memory_cost_estimate'].mean()
        ],
        'Task 4 (Time Pressure)': [
            task4_data['L2_additive_ops_count'].mean(),
            task4_data['L2_subtractive_ops_count'].mean(),
            task4_data['L3_is_rewrite'].sum() / len(task4_data) * 100,
            task4_data['L3_is_minimal'].sum() / len(task4_data) * 100,
            task4_data['L1_edit_distance'].mean(),
            task4_data['memory_cost_estimate'].mean()
        ]
    }, index=['Additive Ops', 'Subtractive Ops', '% Complete Rewrite',
              '% Minimal Change', 'Edit Distance', 'Memory Cost'])

    x = np.arange(len(metrics_comparison2))

    ax.barh(x - width/2, metrics_comparison2['Task 3 (No Restrictions)'], width,
            label='Task 3 (No Restrictions)', color='C2', alpha=0.7)
    ax.barh(x + width/2, metrics_comparison2['Task 4 (Time Pressure)'], width,
            label='Task 4 (Time Pressure)', color='C3', alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(metrics_comparison2.index)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Time Pressure Constraint Effect (Tasks 3 → 4)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, metric in enumerate(metrics_comparison2.index):
        val3 = metrics_comparison2.loc[metric, 'Task 3 (No Restrictions)']
        val4 = metrics_comparison2.loc[metric, 'Task 4 (Time Pressure)']
        ax.text(val3, i - width/2, f' {val3:.2f}', va='center', fontsize=9)
        ax.text(val4, i + width/2, f' {val4:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fig6_constraint_effects.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    return metrics_comparison, metrics_comparison2


# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(df, strategy_counts, constraint_metrics):
    """
    Generate comprehensive summary statistics text file.
    """
    print("\n" + "="*70)
    print("5. Generating Summary Statistics")
    print("="*70)

    with open(SUMMARY_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("Password Editing Behavior Study\n")
        f.write("="*70 + "\n\n")

        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total edits: {len(df)}\n")
        f.write(f"Participants: {df['person_id'].nunique()}\n")
        f.write(f"Tasks: {sorted(df['Task'].unique())}\n")
        f.write(f"Edits per task: {df.groupby('Task').size().to_dict()}\n\n")

        # Strategy distribution
        f.write("STRATEGY DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        f.write("Top strategies (overall):\n")
        total = len(df)
        cumsum = 0
        for i, (strategy, count) in enumerate(strategy_counts.items(), 1):
            pct = 100 * count / total
            cumsum += pct
            f.write(f"  {i}. {strategy:25s}: {count:3d} ({pct:5.2f}%)  [Cumulative: {cumsum:5.2f}%]\n")
            if i >= 10:
                break

        f.write("\nStrategy distribution by task:\n")
        for task in [1, 2, 3, 4]:
            f.write(f"\n  Task {task}:\n")
            task_data = df[df['Task'] == task]
            task_strategies = task_data['L3_primary_strategy'].value_counts().head(5)
            for strategy, count in task_strategies.items():
                pct = 100 * count / len(task_data)
                f.write(f"    {strategy:25s}: {count:3d} ({pct:5.2f}%)\n")

        # Operation bias
        f.write("\n\nOPERATION BIAS ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write("Average operations per edit by task:\n")
        ops_by_task = df.groupby('Task')[['L2_additive_ops_count',
                                           'L2_subtractive_ops_count']].mean()
        for task in [1, 2, 3, 4]:
            add = ops_by_task.loc[task, 'L2_additive_ops_count']
            sub = ops_by_task.loc[task, 'L2_subtractive_ops_count']
            f.write(f"  Task {task}: Additive={add:.3f}, Subtractive={sub:.3f}, Ratio={add/sub if sub > 0 else float('inf'):.2f}\n")

        f.write("\nOperation bias distribution (%):\n")
        bias_dist = df.groupby('Task')['L2_operation_bias'].value_counts(normalize=True) * 100
        for task in [1, 2, 3, 4]:
            f.write(f"  Task {task}:\n")
            for bias_type in ['additive', 'subtractive', 'neutral', 'mixed']:
                if (task, bias_type) in bias_dist.index:
                    f.write(f"    {bias_type:15s}: {bias_dist[(task, bias_type)]:5.2f}%\n")

        # Constraint effects
        f.write("\n\nCONSTRAINT EFFECTS\n")
        f.write("-"*70 + "\n")

        f.write("Deletion-only constraint (Tasks 1 → 2):\n")
        metrics1, metrics2 = constraint_metrics
        f.write(metrics1.to_string())
        f.write("\n\n")

        f.write("Time pressure constraint (Tasks 3 → 4):\n")
        f.write(metrics2.to_string())
        f.write("\n\n")

        # Security and memory
        f.write("\n\nSECURITY AND MEMORY METRICS\n")
        f.write("-"*70 + "\n")
        security_by_task = df.groupby('Task')[['security_guesses_log10',
                                                'memory_cost_estimate']].mean()
        f.write("Average metrics by task:\n")
        f.write(security_by_task.to_string())
        f.write("\n\n")

        # Key findings
        f.write("\n\nKEY FINDINGS\n")
        f.write("-"*70 + "\n")

        task2_add = ops_by_task.loc[2, 'L2_additive_ops_count']
        f.write(f"1. DELETION-ONLY CONSTRAINT IS EFFECTIVE:\n")
        f.write(f"   - Task 2 additive operations: {task2_add:.3f} (essentially 0)\n")
        f.write(f"   - Task 2 strategies: {(df[df['Task']==2]['L3_primary_strategy']=='targeted_deletion').sum()}/100 targeted deletions\n\n")

        top5_strategies = strategy_counts.head(5)
        top5_coverage = 100 * top5_strategies.sum() / len(df)
        f.write(f"2. TOP 5 STRATEGIES COVER {top5_coverage:.1f}% OF DATA:\n")
        for strategy, count in top5_strategies.items():
            f.write(f"   - {strategy}\n")
        f.write("\n")

        overall_bias = df['L2_operation_bias'].value_counts()
        f.write(f"3. OVERALL OPERATION BIAS:\n")
        for bias, count in overall_bias.items():
            pct = 100 * count / len(df)
            f.write(f"   - {bias}: {count} ({pct:.1f}%)\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Saved: {SUMMARY_FILE}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all EDA analyses.
    """
    print("\n")
    print("#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  STEP 2: EXPLORATORY DATA ANALYSIS".center(68) + "#")
    print("#" + "  Password Editing Behavior Study".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print("\n")

    # Load data
    df = load_data()

    # Run analyses
    plot_strategy_distributions(df)
    strategy_counts = analyze_strategy_coverage(df)

    plot_security_memory_tradeoff(df)

    plot_operation_bias(df)

    constraint_metrics = analyze_constraint_effects(df)

    generate_summary_statistics(df, strategy_counts, constraint_metrics)

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")

    print("\n" + "="*70)
    print("NEXT STEPS: Use these insights to build the cognitive model (Step 3)")
    print("="*70)
    print("\nKey insights for modeling:")
    print("  1. Focus on top 5-7 strategies (cover 80%+ of data)")
    print("  2. Model λ_constraint effect (Task 2 forces β_add ≈ 0)")
    print("  3. Model λ_time effect (Task 4 may increase noise or shift strategy)")
    print("  4. Consider security-memory tradeoff in utility function")
    print("\n")


if __name__ == "__main__":
    main()
