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
