import requests
import pandas as pd


# function to fetch data
def fetch_indicator_data(indicator_code):
    base_url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    response = requests.get(base_url)

    if response.status_code == 200:
        data = response.json()['value']
        return pd.DataFrame(data)
    else:
        print(f"Failed to fetch data for {indicator_code}")
        return pd.DataFrame()


indicators = {
    "Incidence of tuberculosis": "MDG_0000000020",
    "Number of incident tuberculosis cases": "TB_e_inc_num",
    "Prevalence of tuberculosis": "MDG_0000000023",
    "Tuberculosis detection rate under DOTS": "MDG_0000000022",
    "Tuberculosis treatment success under DOTS": "MDG_0000000024",
    "Number of reported cases of tuberculosis": "WHS3_522",
    "Tuberculosis treatment coverage": "TB_1",
    "Number of incident TB HIV-positive cases": "TB_e_inc_tbhiv_num",
    "Treatment success rate for new TB cases": "TB_c_new_tsr",
    "Treatment success rate for HIV-positive TB cases": "TB_c_tbhiv_tsr",
    "Tuberculosis effective treatment coverage": "TB_effective_treatment_coverage"
}

data_frames = []

for indicator_name, indicator_code in indicators.items():
    df = fetch_indicator_data(indicator_code)
    if not df.empty:
        # Adding a column for the indicator name
        df['Indicator'] = indicator_name
        data_frames.append(df)

# Combine all the fetched data into one DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)

print(combined_df.head())

# Selecting Relevant Columns as the WHO API usually returns a lot of metadata. so I will Focus on the key columns:
combined_df = combined_df[['Indicator',
                           'SpatialDim', 'TimeDim', 'NumericValue']]
combined_df.columns = ['Indicator', 'Country', 'Year', 'Value']

# Pivoting the data so that each indicator becomes a column
final_df = combined_df.pivot_table(
    index=['Country', 'Year'],
    columns='Indicator',
    values='Value'
).reset_index()

# saving the csv
final_df.to_csv('tb_incidence_data.csv', index=False)
print("Data saved to tb_incidence_data.csv")
