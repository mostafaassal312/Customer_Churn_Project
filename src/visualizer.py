import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(fig, filename):
    """
    Helper function to save plots to the reports/figures directory.
    """
    # Ensure directory exists
    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save path
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    print(f"[INFO] Plot saved: {path}")
    plt.close(fig) # Close plot to free memory

def plot_churn_distribution(df):
    """
    Saves a pie chart of Churn vs Non-Churn.
    """
    fig = plt.figure(figsize=(6, 6))
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
    plt.title('Overall Churn Distribution')
    plt.ylabel('')
    save_plot(fig, 'churn_distribution.png')

def plot_contract_impact(df):
    """
    Saves a bar chart showing Churn by Contract Type.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
    plt.title('Churn Rate by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Number of Customers')
    
    # Get current figure to save it
    fig = plt.gcf() 
    save_plot(fig, 'contract_impact.png')

def plot_charges_risk(df):
    """
    Saves a distribution plot for Monthly Charges.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette='coolwarm', element="step")
    plt.title('Churn Risk by Monthly Charges')
    
    fig = plt.gcf()
    save_plot(fig, 'charges_risk_distribution.png')

def generate_all_plots(df):
    """
    Master function to run all plots.
    """
    print("[INFO] Generating analytical figures...")
    
    # Set style
    sns.set_style("whitegrid")
    
    plot_churn_distribution(df)
    plot_contract_impact(df)
    plot_charges_risk(df)
    
    print("[SUCCESS] All figures generated in reports/figures/")