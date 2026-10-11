# Data Science Essentials: Customer Analytics Project

A practical data exploration and analysis project demonstrating statistical summaries, visualizations, correlation analysis, and structured reporting using real-world customer data.

## Project Overview

This project analyzes a customer dataset to:
- Generate statistical summaries of customer demographics and behavior
- Create visualizations for key metrics (age distribution, spending patterns, etc.)
- Identify correlations between variables (e.g., order value vs. churn)
- Determine key factors influencing customer churn
- Present insights in a structured report

## Key Features

✅ Statistical summaries (descriptive statistics, distributions, frequencies)
✅ Visualizations (histograms, box plots, bar charts, correlation heatmaps)
✅ Correlation analysis (Pearson, Spearman, and key insights)
✅ Key influencing factor identification
✅ Structured report generation (markdown + JSON)
✅ Analytical thinking and data exploration skills development

## Project Structure

```
data-science-essentials/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw_customers.csv     # Raw dataset
│   └── clean_customers.csv   # Cleaned dataset
├── src/
│   └── analysis.py           # Main analysis script
├── reports/
│   ├── eda_report.json       # Structured JSON report
│   └── eda_summary.md        # Human-readable summary
├── plots/
│   ├── distribution_plots.png
│   ├── correlation_heatmap.png
│   ├── churn_analysis.png
│   └── summary_stacked_bar.png
└── .gitignore                # Git ignore rules
```

## How to Run

1. Activate the virtual environment:
   ```bash
   cd /c/New D/project/data-science-essentials
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python src/analysis.py
   ```

4. View results:
   - Visualizations: `plots/*.png`
   - Reports: `reports/eda_report.json` and `reports/eda_summary.md`

## Expected Outcomes

- Develop analytical thinking through real data exploration
- Apply statistical methods to understand customer behavior
- Practice data visualization techniques
- Learn to derive actionable business insights from data
- Build a structured analytical workflow

## Data Dictionary

| Column | Description |
|--------|-------------|
| customer_id | Unique customer identifier |
| signup_month | Month of customer signup (numeric) |
| country | Country code (1-8, encoded) |
| gender | Gender (0 = not specified, 1 = female, 2 = male) |
| age | Customer age in years |
| device | Device type (0 = other, 1 = desktop, 2 = tablet, 3 = mobile) |
| plan | Subscription plan (0 = basic, 1 = pro, 2 = enterprise) |
| order_value | Average order value in currency |
| discount_applied | Average discount applied |
| num_items | Average number of items per order |
| payment_method | Payment method (0 = other, 1 = credit card, 2 = PayPal, 3 = bank transfer) |
| churned | Churn status (True/False) |
| review_score | Customer review score (0-5) |
| is_premium | Premium customer flag (0 = no, 1 = yes) |
| high_value | High-value customer flag (0 = no, 1 = yes) |

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (optional, for ML extension)

## License

MIT License - feel free to use and modify.
