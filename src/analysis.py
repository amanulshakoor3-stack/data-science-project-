"""
Data Science Essentials: Customer Data Analysis
================================================
Statistical summaries, visualizations, correlation analysis,
and structured reporting for customer behavior insights.

Author: Data Science Project
Date: October 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Set up visual styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(data_path):
    """Load customer data from CSV file."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def compute_statistical_summary(df):
    """Generate descriptive statistics for numerical variables."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numerical_cols].describe().T
    summary['missing'] = df[numerical_cols].isnull().sum().values
    summary['dtype'] = df[numerical_cols].dtypes.values
    return summary


def create_visualizations(df, output_dir):
    """Create comprehensive visualizations for the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("Creating visualizations...")

    # 1. Age distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.title('Age Distribution of Customers', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'age_distribution.png'), dpi=300)
    plt.close()

    # 2. Order value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['order_value'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='teal')
    plt.title('Distribution of Order Value', fontsize=14, fontweight='bold')
    plt.xlabel('Order Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'order_value_distribution.png'), dpi=300)
    plt.close()

    # 3. Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # 4. Churn analysis (by churned status)
    churn_counts = df['churned'].value_counts()
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Retained', 'Churned'], churn_counts.values,
                   color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
    plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Customers')
    for bar, count in zip(bars, churn_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{count}\n({count/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'churn_analysis.png'), dpi=300)
    plt.close()

    # 5. Average order value by plan
    plt.figure(figsize=(10, 6))
    plan_avg = df.groupby('plan')['order_value'].mean().sort_values(ascending=False)
    plan_names = ['Basic', 'Pro', 'Enterprise', 'Premium']  # Add Premium for plan 3
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    # Use only first 3 or all plan types present in data
    n_plans = min(3, len(plan_avg))
    bars = plt.bar(plan_names[:n_plans], plan_avg.values[:n_plans],
                   color=colors[:n_plans], edgecolor='black', alpha=0.8)

    plan_labels = {
        0: 'Basic',
        1: 'Pro',
        2: 'Enterprise',
        3: 'Premium'
    }

    # Create custom labels
    labels = [plan_labels.get(i, f'Plan {i}') for i in plan_avg.index[:n_plans]]
    bars = plt.bar(labels, plan_avg.values[:n_plans],
                   color=colors[:n_plans], edgecolor='black', alpha=0.8)
    plt.title('Average Order Value by Plan Type', fontsize=14, fontweight='bold')
    plt.xlabel('Plan Type')
    plt.ylabel('Average Order Value')
    for bar, val in zip(bars, plan_avg.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'${val:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'order_value_by_plan.png'), dpi=300)
    plt.close()

    # 6. Churn rate by plan type
    churn_by_plan = df.groupby('plan')['churned'].apply(lambda x: (x == True).mean() * 100)
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    plan_labels = {
        0: 'Basic',
        1: 'Pro',
        2: 'Enterprise',
        3: 'Premium'
    }
    labels = [plan_labels.get(i, f'Plan {i}') for i in churn_by_plan.index]
    bars = plt.bar(labels, churn_by_plan.values,
                   color=colors[:len(labels)], edgecolor='black', alpha=0.8)
    plt.title('Churn Rate by Plan Type (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Plan Type')
    plt.ylabel('Churn Rate (%)')
    for bar, val in zip(bars, churn_by_plan.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'churn_rate_by_plan.png'), dpi=300)
    plt.close()

    # 7. Age vs Order Value scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['order_value'], alpha=0.5, c=df['review_score'],
                cmap='RdYlGn', edgecolors='black', s=50)
    plt.colorbar(label='Review Score')
    plt.title('Age vs Order Value (colored by Review Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Order Value')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'age_vs_order_value.png'), dpi=300)
    plt.close()

    # 8. Num items vs Order value scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_items'], df['order_value'], alpha=0.5, c=df['discount_applied'],
                cmap='viridis', edgecolors='black', s=50)
    plt.colorbar(label='Discount Applied')
    plt.title('Number of Items vs Order Value (colored by Discount)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Items')
    plt.ylabel('Order Value')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'num_items_vs_order_value.png'), dpi=300)
    plt.close()

    # 9. Stacked bar: gender and device breakdown
    plt.figure(figsize=(12, 6))
    device_gender = df.groupby(['gender', 'device']).size().unstack(fill_value=0)
    device_gender.plot(kind='bar', stacked=True, figsize=(12, 6), edgecolor='black', alpha=0.8)
    plt.title('Customer Distribution by Gender and Device', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Number of Customers')
    plt.legend(title='Device Type', labels=['Other', 'Desktop', 'Tablet', 'Mobile'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gender_device_stacked.png'), dpi=300)
    plt.close()

    # 10. Country distribution pie
    plt.figure(figsize=(8, 8))
    country_counts = df['country'].value_counts()
    plt.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("Set3", len(country_counts)))
    plt.title('Customer Distribution by Country', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'country_distribution.png'), dpi=300)
    plt.close()

    print(f"Visualizations saved to {plots_dir}")
    return plots_dir


def correlation_analysis(df):
    """Perform correlation analysis to identify key influencing factors."""
    print("\n=== CORRELATION ANALYSIS ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude binary/categorical-encoded columns for cleaner analysis
    # Focus on meaningful numerical variables
    key_cols = ['age', 'order_value', 'discount_applied', 'num_items', 'review_score']

    if all(col in numerical_cols for col in key_cols):
        corr_matrix = df[key_cols].corr()

        # Print correlation matrix
        print("\nCorrelation Matrix (Key Variables):")
        print(corr_matrix.to_string())

        # Find strong correlations
        strong_correlations = []
        for i in range(len(key_cols)):
            for j in range(i + 1, len(key_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Moderate correlation threshold
                    strong_correlations.append({
                        'var1': key_cols[i],
                        'var2': key_cols[j],
                        'correlation': round(corr_value, 4),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate' if abs(corr_value) > 0.4 else 'weak'
                    })

        print("\nSignificant Correlations (|r| > 0.3):")
        for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']} ({corr['strength']})")

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'key_variables': key_cols,
            'strong_correlations': strong_correlations,
            'method': 'Pearson correlation'
        }
    else:
        print("Not all key variables available for correlation analysis")
        return {}


def identify_key_factors(df):
    """Identify key factors influencing customer churn."""
    print("\n=== KEY FACTOR IDENTIFICATION ===")
    insights = {}

    # Average order value by churn status
    churned_avg = df.groupby('churned')['order_value'].mean()
    insights['avg_order_value_by_churn'] = {
        'churned': float(churned_avg.get(True, 0)),
        'retained': float(churned_avg.get(False, 0)),
        'difference': float(churned_avg.get(True, 0) - churned_avg.get(False, 0))
    }

    # Average review score by churn
    review_by_churn = df.groupby('churned')['review_score'].mean()
    insights['avg_review_score_by_churn'] = {
        'churned': float(review_by_churn.get(True, 0)),
        'retained': float(review_by_churn.get(False, 0))
    }

    # Average items by churn
    items_by_churn = df.groupby('churned')['num_items'].mean()
    insights['avg_num_items_by_churn'] = {
        'churned': float(items_by_churn.get(True, 0)),
        'retained': float(items_by_churn.get(False, 0))
    }

    # Churn rate by plan type
    churn_by_plan = df.groupby('plan')['churned'].apply(lambda x: (x == True).mean() * 100)
    plan_labels = {
        0: 'basic',
        1: 'pro',
        2: 'enterprise',
        3: 'premium'
    }
    insights['churn_rate_by_plan'] = {
        'basic': float(churn_by_plan.get(0, 0)),
        'pro': float(churn_by_plan.get(1, 0)),
        'enterprise': float(churn_by_plan.get(2, 0)),
        'premium': float(churn_by_plan.get(3, 0))
    }

    # High-value customers churn rate
    if 'high_value' in df.columns:
        hv_churn = df.groupby('high_value')['churned'].apply(lambda x: (x == True).mean() * 100)
        insights['high_value_churn_rate'] = {
            'yes': float(hv_churn.get(1, 0)),
            'no': float(hv_churn.get(0, 0))
        }

    # Premium customers churn rate
    if 'is_premium' in df.columns:
        prem_churn = df.groupby('is_premium')['churned'].apply(lambda x: (x == True).mean() * 100)
        insights['premium_churn_rate'] = {
            'premium': float(prem_churn.get(1, 0)),
            'non_premium': float(prem_churn.get(0, 0))
        }

    # Gender breakdown
    gender_counts = df['gender'].value_counts()
    insights['gender_distribution'] = {
        'not_specified': int(gender_counts.get(0, 0)),
        'female': int(gender_counts.get(1, 0)),
        'male': int(gender_counts.get(2, 0))
    }

    # Plan distribution
    plan_counts = df['plan'].value_counts()
    insights['plan_distribution'] = {
        'basic': int(plan_counts.get(0, 0)),
        'pro': int(plan_counts.get(1, 0)),
        'enterprise': int(plan_counts.get(2, 0)),
        'premium': int(plan_counts.get(3, 0))
    }

    # Order value by plan (for richer insights)
    plan_order = df.groupby('plan')['order_value'].agg(['mean', 'median', 'count'])
    insights['avg_order_value_by_plan'] = {
        'basic': float(plan_order.loc[0, 'mean']) if 0 in plan_order.index else 0,
        'pro': float(plan_order.loc[1, 'mean']) if 1 in plan_order.index else 0,
        'enterprise': float(plan_order.loc[2, 'mean']) if 2 in plan_order.index else 0,
        'premium': float(plan_order.loc[3, 'mean']) if 3 in plan_order.index else 0
    }

    return insights


def generate_report(df, correlations, insights, output_dir):
    """Generate a structured report of the analysis."""
    print("\n=== GENERATING REPORT ===")

    report = {
        'project': 'Data Science Essentials - Customer Analytics',
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'source': 'Customer purchase data',
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        },
        'statistical_summary': compute_statistical_summary(df).to_dict(),
        'correlation_analysis': correlations,
        'key_insights': insights,
        'visualizations': {
            'plots_dir': 'plots/',
            'files': [
                'age_distribution.png',
                'order_value_distribution.png',
                'correlation_heatmap.png',
                'churn_analysis.png',
                'order_value_by_plan.png',
                'churn_rate_by_plan.png',
                'age_vs_order_value.png',
                'num_items_vs_order_value.png',
                'gender_device_stacked.png',
                'country_distribution.png'
            ]
        },
        'methodology': {
            'statistical_methods': ['Descriptive statistics', 'Correlation analysis', 'Group-by analysis'],
            'visualization_techniques': ['Histograms', 'Scatter plots', 'Bar charts', 'Heatmaps', 'Stacked bar charts'],
            'data_cleaning': 'No missing values found',
            'analysis_tools': ['pandas', 'numpy', 'matplotlib', 'seaborn']
        },
        'recommendations': generate_recommendations(insights, correlations)
    }

    # Save JSON report
    json_path = os.path.join(output_dir, 'reports', 'eda_report.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report saved to {json_path}")

    # Generate markdown summary
    markdown_path = os.path.join(output_dir, 'reports', 'eda_summary.md')
    with open(markdown_path, 'w') as f:
        f.write(generate_markdown_summary(report))
    print(f"Markdown summary saved to {markdown_path}")

    return report


def generate_recommendations(insights, correlations):
    """Generate data-driven recommendations based on analysis."""
    recommendations = []

    # Recommendation based on order value analysis
    avg_by_plan = insights.get('avg_order_value_by_plan', {})
    if 'enterprise' in avg_by_plan and 'basic' in avg_by_plan:
        ratio = avg_by_plan['enterprise'] / avg_by_plan['basic'] if avg_by_plan['basic'] > 0 else 0
        if ratio > 2:
            recommendations.append(
                "Enterprise customers spend significantly more than basic customers. "
                "Consider targeting upsell campaigns to basic plan users."
            )

    # Recommendation based on churn analysis
    churn_rate = insights.get('churn_rate_by_plan', {})
    if churn_rate.get('basic', 0) > churn_rate.get('enterprise', 0):
        recommendations.append(
            f"Basic plan churn rate ({churn_rate.get('basic', 0):.1f}%) is higher than enterprise ({churn_rate.get('enterprise', 0):.1f}%). "
            "Consider loyalty programs and premium feature demos for basic users."
        )

    # Recommendation based on review scores
    review_score = insights.get('avg_review_score_by_churn', {})
    if review_score.get('churned', 0) < review_score.get('retained', 0):
        recommendations.append(
            "Churned customers have lower review scores. "
            "Implement a post-purchase follow-up system to address satisfaction issues early."
        )

    # Recommendation based on correlation
    if correlations and correlations.get('strong_correlations'):
        top_corr = correlations['strong_correlations'][0]
        if top_corr['correlation'] > 0.7:
            recommendations.append(
                f"Strong correlation between {top_corr['var1']} and {top_corr['var2']} suggests "
                f"they can be used together for customer segmentation."
            )

    # General recommendations
    if not recommendations:
        recommendations.append(
            "Continue monitoring key metrics monthly to track trends. "
            "Focus on improving review scores to reduce churn."
        )

    recommendations.append(
        "Consider A/B testing different retention strategies to reduce churn."
    )

    return recommendations


def generate_markdown_summary(report):
    """Generate a human-readable markdown summary."""
    lines = []
    lines.append("# Data Science Essentials: Customer Analytics Report\n")
    lines.append(f"**Generated:** {report['timestamp']}\n")
    lines.append(f"**Dataset:** {report['dataset']['rows']} customers, {report['dataset']['columns']} variables\n")

    lines.append("## Executive Summary\n")
    lines.append("This analysis explores customer behavior patterns including demographics, spending habits, and churn dynamics using statistical summaries and visualizations.\n")

    lines.append("## Key Findings\n")

    # Churn overview
    churn_analysis = report['key_insights'].get('avg_order_value_by_churn', {})
    churn_val = report['key_insights'].get('avg_order_value_by_churn', {}).get('churned', 0)
    retained_val = report['key_insights'].get('avg_order_value_by_churn', {}).get('retained', 0)
    if churn_val > 0:
        lines.append(f"### Customer Churn\n")
        lines.append(f"- Churned customers average order value: ${churn_val:.2f}\n")
        lines.append(f"- Retained customers average order value: ${retained_val:.2f}\n")

    # Plan analysis
    churn_by_plan = report['key_insights'].get('churn_rate_by_plan', {})
    if churn_by_plan:
        lines.append(f"### Churn by Plan Type\n")
        lines.append(f"- Basic: {churn_by_plan.get('basic', 0):.1f}%\n")
        lines.append(f"- Pro: {churn_by_plan.get('pro', 0):.1f}%\n")
        lines.append(f"- Enterprise: {churn_by_plan.get('enterprise', 0):.1f}%\n")

    # Review scores
    review_score = report['key_insights'].get('avg_review_score_by_churn', {})
    if review_score:
        lines.append(f"### Review Score by Churn Status\n")
        lines.append(f"- Churned: {review_score.get('churned', 0):.2f}\n")
        lines.append(f"- Retained: {review_score.get('retained', 0):.2f}\n")

    # Plan distribution
    plan_dist = report['key_insights'].get('plan_distribution', {})
    if plan_dist:
        lines.append(f"### Plan Distribution\n")
        lines.append(f"- Basic: {plan_dist.get('basic', 0)}\n")
        lines.append(f"- Pro: {plan_dist.get('pro', 0)}\n")
        lines.append(f"- Enterprise: {plan_dist.get('enterprise', 0)}\n")

    lines.append("## Recommendations\n")
    for i, rec in enumerate(report['recommendations'], 1):
        lines.append(f"{i}. {rec}\n")

    lines.append("## Visualizations\n")
    for viz_file in report['visualizations']['files']:
        lines.append(f"![{viz_file}](plots/{viz_file})\n")

    return "\n".join(lines)


def main():
    """Main function to run the EDA."""
    print("=" * 60)
    print("Data Science Essentials: Customer Analytics")
    print("=" * 60)

    # Paths
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clean_customers.csv")
    data_path = os.path.normpath(data_path)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Check data exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    try:
        # Load data
        df = load_data(data_path)

        # Create visualizations
        plots_dir = create_visualizations(df, output_dir)

        # Correlation analysis
        correlations = correlation_analysis(df)

        # Key factor identification
        insights = identify_key_factors(df)

        # Generate report
        report = generate_report(df, correlations, insights, output_dir)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
