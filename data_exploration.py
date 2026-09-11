"""
Data Exploration Script
Analyze and visualize the credit card fraud dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    return df

def basic_statistics(df):
    """Display basic statistics"""
    print("\n" + "="*50)
    print("BASIC STATISTICS")
    print("="*50)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")

def class_distribution(df):
    """Analyze class distribution"""
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    
    class_counts = df['Class'].value_counts()
    print(f"\nClass Distribution:\n{class_counts}")
    
    percentages = df['Class'].value_counts(normalize=True) * 100
    print(f"\nClass Percentages:\n{percentages}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution - Fraud vs Normal')
    plt.xlabel('Class (0=Normal, 1=Fraud)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nClass distribution plot saved to results/class_distribution.png")
    plt.close()

def amount_analysis(df):
    """Analyze transaction amounts"""
    print("\n" + "="*50)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("="*50)
    
    print(f"\nAmount Statistics:\n{df['Amount'].describe()}")
    
    # Separate fraud and normal
    fraud_amounts = df[df['Class'] == 1]['Amount']
    normal_amounts = df[df['Class'] == 0]['Amount']
    
    print(f"\nFraud Amount Stats:\n{fraud_amounts.describe()}")
    print(f"\nNormal Amount Stats:\n{normal_amounts.describe()}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist([normal_amounts, fraud_amounts], bins=50, label=['Normal', 'Fraud'], color=['green', 'red'])
    axes[0].set_xlabel('Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Transaction Amount Distribution')
    axes[0].legend()
    
    axes[1].boxplot([normal_amounts, fraud_amounts], labels=['Normal', 'Fraud'])
    axes[1].set_ylabel('Amount')
    axes[1].set_title('Transaction Amount Box Plot')
    
    plt.tight_layout()
    plt.savefig('results/amount_analysis.png', dpi=300, bbox_inches='tight')
    print("Amount analysis plot saved to results/amount_analysis.png")
    plt.close()

def time_analysis(df):
    """Analyze transaction time"""
    print("\n" + "="*50)
    print("TRANSACTION TIME ANALYSIS")
    print("="*50)
    
    print(f"\nTime Statistics:\n{df['Time'].describe()}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    fraud_times = df[df['Class'] == 1]['Time']
    normal_times = df[df['Class'] == 0]['Time']
    
    plt.scatter(normal_times, [0]*len(normal_times), alpha=0.5, label='Normal', s=10)
    plt.scatter(fraud_times, [1]*len(fraud_times), alpha=0.5, label='Fraud', s=10)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Class')
    plt.title('Transaction Time Distribution')
    plt.yticks([0, 1], ['Normal', 'Fraud'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/time_analysis.png', dpi=300, bbox_inches='tight')
    print("Time analysis plot saved to results/time_analysis.png")
    plt.close()

def correlation_analysis(df):
    """Analyze feature correlations"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Features only (exclude Time and Amount for initial analysis)
    features = [col for col in df.columns if col.startswith('V')]
    correlations = df[features + ['Class']].corr()['Class'].drop('Class').sort_values(ascending=False)
    
    print("\nTop 10 Features Correlated with Fraud:")
    print(correlations.head(10))
    
    # Full correlation heatmap
    plt.figure(figsize=(16, 14))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, annot=False)
    plt.title('Correlation Matrix - All Features')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix saved to results/correlation_matrix.png")
    plt.close()

def feature_distributions(df):
    """Analyze feature distributions"""
    print("\n" + "="*50)
    print("FEATURE DISTRIBUTIONS")
    print("="*50)
    
    # Sample features for visualization
    features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V5']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        fraud_data = df[df['Class'] == 1][feature]
        normal_data = df[df['Class'] == 0][feature]
        
        axes[idx].hist([normal_data, fraud_data], bins=50, label=['Normal', 'Fraud'], color=['green', 'red'], alpha=0.7)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].legend()
    
    # Remove extra subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('results/feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Feature distributions saved to results/feature_distributions.png")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "="*50)
    print("DATA EXPLORATION - CREDIT CARD FRAUD")
    print("="*50)
    
    df = load_data('data/creditcard.csv')
    
    basic_statistics(df)
    class_distribution(df)
    amount_analysis(df)
    time_analysis(df)
    feature_distributions(df)
    correlation_analysis(df)
    
    print("\n" + "="*50)
    print("EXPLORATION COMPLETED!")
    print("="*50)
    print("\nGenerated visualizations:")
    print("  - results/class_distribution.png")
    print("  - results/amount_analysis.png")
    print("  - results/time_analysis.png")
    print("  - results/feature_distributions.png")
    print("  - results/correlation_matrix.png")

if __name__ == "__main__":
    main()
