import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_plant_data():
    data_dir = "/Users/sk/Desktop/proj"
    weather_file = f"{data_dir}/Plant_1_Weather_Sensor_Data.csv"
    gen_file = f"{data_dir}/Plant_1_Generation_Data.csv"
    
    df_weather = pd.read_csv(weather_file)
    df_gen = pd.read_csv(gen_file)
    
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
    
    df_weather.set_index('DATE_TIME', inplace=True)
    df_gen.set_index('DATE_TIME', inplace=True)
    
    # Resample to Hourly
    df_weather_hourly = df_weather.resample('h').mean(numeric_only=True)
    df_gen_hourly = df_gen.resample('h').mean(numeric_only=True)
    
    # Merge (suffix for overlapping columns)
    df = df_weather_hourly.join(df_gen_hourly, how='inner', lsuffix='_weather', rsuffix='_gen')
    
    # Add engineered features
    df['hour'] = df.index.hour
    df['GHI_x_Temp'] = df['IRRADIATION'] * df['AMBIENT_TEMPERATURE']
    df['GHI_sq'] = df['IRRADIATION'] ** 2
    
    print("Correlation with AC_POWER:")
    corr = df.corr()['AC_POWER'].sort_values(ascending=False)
    print(corr)
    
    # Check linearity
    plt.figure(figsize=(10, 6))
    plt.scatter(df['IRRADIATION'], df['AC_POWER'], alpha=0.5)
    plt.title("Irradiation vs AC Power")
    plt.xlabel("Irradiation")
    plt.ylabel("AC Power")
    plt.savefig("artifacts/plant_correlation.png")

if __name__ == "__main__":
    analyze_plant_data()
