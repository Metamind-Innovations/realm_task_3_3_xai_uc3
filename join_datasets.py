import pandas as pd

patient_df = pd.read_csv('dataset_small/patient_age_diagnosis.csv')
glucose_df = pd.read_csv('dataset_small/glucose_insulin_100K.csv')

print("Patient DataFrame shape:", patient_df.shape)
print("Glucose DataFrame shape:", glucose_df.shape)

patient_df.columns = [col.lower() for col in patient_df.columns]
glucose_df.columns = [col.lower() for col in glucose_df.columns]

if 'iucstay_id' in patient_df.columns:
    patient_df = patient_df.rename(columns={'iucstay_id': 'icustay_id'})

patient_key_counts = patient_df.groupby(['subject_id', 'hadm_id', 'icustay_id']).size()
print(f"\nPatients with multiple diagnoses: {sum(patient_key_counts > 1)}")
print(f"Maximum diagnoses per patient: {patient_key_counts.max()}")

print("\nApproach: Grouping multiple diagnoses into lists")
# Group diagnoses (ICD9 codes) for each unique patient encounter
patient_grouped = patient_df.groupby(['subject_id', 'hadm_id', 'icustay_id']).agg({
    'icd9_code': lambda x: ','.join(sorted(set(x.astype(str)))),
    'gender': 'first',
    'admission_age': 'first'
}).reset_index()

print(f"Patient DataFrame after grouping: {patient_grouped.shape}")

joined_df = pd.merge(
    patient_grouped,
    glucose_df,
    on=['subject_id', 'hadm_id', 'icustay_id'],
    how='inner',
    validate='one_to_many'
)

print("\nJoined DataFrame information:")
print(f"Number of rows: {len(joined_df)}")
print(f"Number of columns: {len(joined_df.columns)}")
print("\nFirst 5 rows of joined data:")
print(joined_df.head())

duplicate_rows = joined_df.duplicated().sum()
print(f"\nRemaining duplicate rows: {duplicate_rows}")

output_file = 'dataset_small/joined_patient_glucose_data.csv'
joined_df.to_csv(output_file, index=False)
print(f"\nSuccessfully joined data saved to '{output_file}'")
