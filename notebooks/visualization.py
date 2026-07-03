"""
Visualization Module for Research Project
Generates all figures and charts for the paper
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

RESULTS_DIR = "results"

def save_fig(name):
    plt.savefig(f"{RESULTS_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/{name}.pdf", bbox_inches='tight')
    plt.close()

def plot_ethical_scores(ethical_df):
    """Figure 1: Ethical scores by scenario type"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Ethical Score distribution
    sns.boxplot(data=ethical_df, x='scenario_type', y='ethical_score', ax=axes[0,0])
    axes[0,0].set_title('Ethical Score Distribution by Scenario')
    axes[0,0].set_ylabel('Ethical Score')

    # Bias Score
    sns.barplot(data=ethical_df.groupby('scenario_type')['bias_score'].mean().reset_index(),
                x='scenario_type', y='bias_score', ax=axes[0,1])
    axes[0,1].set_title('Average Bias Score by Scenario')
    axes[0,1].set_ylabel('Bias Score (normalized)')

    # Explainability
    sns.violinplot(data=ethical_df, x='scenario_type', y='explainability_score', ax=axes[1,0])
    axes[1,0].set_title('Explainability Score Distribution')
    axes[1,0].set_ylabel('Explainability Score')

    # Risk Factor
    sns.barplot(data=ethical_df.groupby('scenario_type')['risk_factor'].mean().reset_index(),
                x='scenario_type', y='risk_factor', ax=axes[1,1])
    axes[1,1].set_title('Average Risk Factor by Scenario')
    axes[1,1].set_ylabel('Risk Factor')

    plt.tight_layout()
    save_fig("fig1_ethical_analysis")

def plot_environmental_footprint(env_df):
    """Figure 2: Environmental footprint comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['energy_kwh', 'water_liters', 'carbon_kgco2e']
    titles = ['Energy Consumption (kWh)', 'Water Usage (L)', 'Carbon Footprint (kgCO₂e)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        pivot = env_df.pivot(index='model', columns='task', values=metric)
        pivot.plot(kind='bar', ax=axes[idx])
        axes[idx].set_title(title)
        axes[idx].set_ylabel(title)
        axes[idx].legend(title='Task')
        axes[idx].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    save_fig("fig2_environmental_footprint")

def plot_rai_scores(rai_df):
    """Figure 3: RAI Score heatmap"""
    plt.figure(figsize=(10, 6))
    pivot = rai_df.pivot(index='scenario_type', columns='model', values='rai_score')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100)
    plt.title('Responsible AI (RAI) Score Heatmap')
    plt.ylabel('Scenario Type')
    plt.xlabel('Model')
    plt.tight_layout()
    save_fig("fig3_rai_heatmap")

def plot_optimization_impact(rec_df):
    """Figure 4: Optimization impact comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rec_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, rec_df['energy_reduction_%'], width, label='Energy Reduction %', color='green')
    bars2 = ax.bar(x + width/2, rec_df['accuracy_loss_%'], width, label='Accuracy Loss %', color='red')

    ax.set_xlabel('Optimization Technique')
    ax.set_ylabel('Percentage')
    ax.set_title('Optimization Techniques: Energy Savings vs Accuracy Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(rec_df['technique'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    save_fig("fig4_optimization_impact")

def plot_pareto_frontier(rai_df):
    """Figure 5: Pareto frontier between ethical and environmental performance"""
    plt.figure(figsize=(10, 7))

    for model in rai_df['model'].unique():
        model_data = rai_df[rai_df['model'] == model]
        plt.scatter(model_data['ethical_score'], model_data['environmental_score'], 
                   label=model, s=100, alpha=0.7)

    plt.xlabel('Ethical Score')
    plt.ylabel('Environmental Score')
    plt.title('Pareto Frontier: Ethical vs Environmental Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("fig5_pareto_frontier")

def plot_water_carbon_comparison():
    """Figure 6: Water vs Carbon across grid regions"""
    regions = ['Iceland\n(Renewable)', 'France\n(Nuclear)', 'US\n(Average)', 'China\n(Coal)', 'India\n(Coal)']
    carbon_intensity = [50, 80, 450, 800, 750]
    water_factor = [0.5, 0.6, 1.0, 1.2, 1.3]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(regions))
    ax1.bar(x - 0.2, carbon_intensity, 0.4, label='Carbon Intensity (gCO₂e/kWh)', color='brown')
    ax1.set_xlabel('Grid Region')
    ax1.set_ylabel('Carbon Intensity (gCO₂e/kWh)', color='brown')
    ax1.tick_params(axis='y', labelcolor='brown')

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, water_factor, 0.4, label='Water Factor (L/kWh)', color='blue')
    ax2.set_ylabel('Water Factor (L/kWh)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.xticks(x, regions)
    plt.title('Environmental Impact Across Grid Regions')
    plt.tight_layout()
    save_fig("fig6_grid_comparison")

def generate_all_visualizations(ethical_df, env_df, rai_df, rec_df):
    """Generate all visualization figures"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_ethical_scores(ethical_df)
    plot_environmental_footprint(env_df)
    plot_rai_scores(rai_df)
    plot_optimization_impact(rec_df)
    plot_pareto_frontier(rai_df)
    plot_water_carbon_comparison()
    print("✅ All 6 figures generated successfully!")
