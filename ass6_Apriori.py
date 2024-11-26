import pandas as pd

def load_data(csv_file): 
    # Read CSV into a pandas DataFrame 
    data = pd.read_csv(csv_file) 
 
    # Print column names for verification 
    print("Columns in CSV:", data.columns) 
 
    # Strip any leading/trailing spaces in column names 
    data.columns = data.columns.str.strip() 
 
    return data

def preprocess_data(df): 
    # Group jobs listed by Company ID 
    grouped = df.groupby('Company_id')['Jobs_listed'].apply(lambda x: set(','.join(x).split(','))) 
 
    # Get the set of all unique jobs in the dataset 
    all_jobs = set(df['Jobs_listed'].str.split(',').sum()) 
 
    return grouped, all_jobs

def companies_with_all_jobs(grouped, all_jobs): 
    # Find companies where the set of jobs listed matches the full set of all jobs 
    companies_with_all = [company_id for company_id, jobs in grouped.items() if jobs >= 
all_jobs] 
    return companies_with_all 

def main(csv_file): 
    # Load and preprocess data 
    data = load_data(csv_file) 
    grouped, all_jobs = preprocess_data(data) 
 
    # Find companies listing all jobs 
    companies = companies_with_all_jobs(grouped, all_jobs) 
 
    print(f"Companies that listed all jobs: {companies}")

if __name__ == "__main__": 
    # Provide your CSV file path here 
    csv_file_path = "job_data.csv"  # Replace with your file path 
    main(csv_file_path)