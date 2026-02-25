# Passenger Traffic Analysis

**Repository Name:** `passenger-traffic-analysis`

## Description
This project analyzes passenger traffic and revenue data for public transport (bus, tram, metro) in 2019 and 2022. It includes data cleaning, preprocessing, visualization of daily passenger trends, Fourier smoothing, and revenue calculation.

## Features
- Load and preprocess CSV datasets from 2019 and 2022.
- Normalize date and numeric columns.
- Build daily passenger series and handle missing values with interpolation.
- Visualize trends with scatter plots, smoothed curves, and revenue annotations.
- Compare transport modes’ revenue across years.
- Save summary figures and revenue text files.

## Files
- `2019data1.csv` – Original passenger data for 2019.
- `2022data1.csv` – Original passenger data for 2022.
- `Figure1.png` – Daily passenger numbers with Fourier smoothing.
- `Figure2.png` – Revenue summary annotations.
- `Figure3.png` / `Figure4.png` – Additional visualizations.
- `revenues_summary.txt` – Summary of revenues by transport mode.

## Usage
1. Place `2019data1.csv` and `2022data1.csv` in the same directory as the script.
2. Run the Python script:
`python passenger_traffic_analysis.py`
3. Outputs (figures and summary text) will be saved in the same folder.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
