import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# Data Generation Function with Additional Features
def generate_synthetic_data(start_date, end_date, num_subnets):
    date_range = pd.date_range(start=start_date, end=end_date)
    data = []

    cidr_sizes = [128, 256]

    for date in date_range:
        for subnet_id in range(1, num_subnets + 1):
            subnet_size = random.choice(cidr_sizes)
            ips_allocated = random.randint(0, int(subnet_size * 0.9))  # 90% of the subnet size
            # Additional feature: average_ips_allocated_last_month
            if date.month == 1:
                avg_ips_allocated_last_month = 0  # No data for the previous month in January
            else:
                previous_month_data = [entry for entry in data if entry[1] == subnet_id and entry[3] == date.month - 1]
                if previous_month_data:
                    avg_ips_allocated_last_month = np.mean([entry[0] for entry in previous_month_data])
                else:
                    avg_ips_allocated_last_month = 0
            data.append([ips_allocated, subnet_id, subnet_size, date.month, date.year, avg_ips_allocated_last_month])
    
    df = pd.DataFrame(data, columns=['ips_allocated', 'subnet_id', 'subnet_size', 'month', 'year', 'avg_ips_allocated_last_month'])
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    df.to_csv('synthetic_train_data.csv', index=False)
    
    df.to_csv('synthetic_data.csv', index=False)

    # Test data without the label
    test_df = df.drop(columns=['ips_allocated'])
    test_df.to_csv('synthetic_test_data.csv', index=False)
    
    return df

# Generate synthetic data
start_date = '2023-01-01'
end_date = '2023-12-31'
num_subnets = 10
df = generate_synthetic_data(start_date, end_date, num_subnets)

# Plot the distribution of the target variable
plt.hist(df['ips_allocated'], bins=30)
plt.title('Distribution of IPs Allocated')
plt.xlabel('IPs Allocated')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Scatter plot matrix
pd.plotting.scatter_matrix(df, figsize=(15, 15))
plt.show()