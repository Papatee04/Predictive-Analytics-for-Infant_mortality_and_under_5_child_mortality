import pandas as pd

# Replace 'your_dta_file.dta' with the actual path to your .dta file
dta_file = 'C:/Users/tlche/Downloads/Compressed/ZMIR61DT/ZMIR61FL.DTA'

# Read the .dta file with convert_categoricals set to False
df = pd.read_stata(dta_file, convert_categoricals=False)

# Save the DataFrame as a CSV file
df.to_csv('output201314.csv', index=False)

print("Conversion complete!")
