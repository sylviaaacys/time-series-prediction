# Household Electricity Load Forecasting

A CS412 final project comparing **LSTM**, **XGBoost**, and **Transformer** models for household electricity load prediction. The study explores how temporal patterns, holidays, weather, and historical load features affect forecasting performance.

**Best reported result:** a three-layer LSTM with temporal, weekly cyclic, and holiday features achieved **RMSE 0.31**, **MAE 0.21**, and **sMAPE 35.84%**.

[Read the final report](FinalProject/CS412_Final_Report.pdf) for the methodology, experiment tables, and prediction plots.

## Dataset

The project uses electricity load measurements from a household in **Jianshan Town, Haining City, China**, covering January 2022 through December 2024. Each day contains **96 measurements at 15-minute intervals**.

The report describes the following splits:

| Split | Period |
| --- | --- |
| Training and testing | January 1, 2022 - December 30, 2023 |
| Validation | January 1, 2024 - May 31, 2024 |
| Checking / prediction | June 1, 2024 - December 1, 2024 |

The Excel datasets are included inside the Transformer ZIP archive. Weather experiments use historical data from Open-Meteo, aligned to the `Asia/Shanghai` timezone.

## Approach

The workflow combines exploratory analysis, missing-value interpolation, feature engineering, model training, and evaluation. Neural-network inputs are scaled with `MinMaxScaler`; XGBoost uses unscaled features.

| Model | Implementation | Main configuration described in the report |
| --- | --- | --- |
| LSTM | TensorFlow / Keras | Sliding windows of four 15-minute observations; two- and three-layer variants; Adam optimizer; MSE loss; early stopping |
| XGBoost | `XGBRegressor` | Tree-based boosting; up to 1,000 trees; maximum depth 10; learning rate 0.01; early stopping |
| Transformer | PyTorch | Seven-day input windows over daily load profiles; sinusoidal positional encoding; three encoder layers; eight attention heads; model dimension 64; AdamW optimizer |

The LSTM and XGBoost experiments investigate:

- **Temporal features:** season, month, hour, year, and weekly sine/cosine encodings.
- **Holiday indicators:** national holidays to capture changes in household routines.
- **Weather features:** daily and hourly temperature, with additional exploratory weather variables.
- **Lag features:** past load values at hourly, daily, weekly, and yearly offsets, plus cyclic hour encodings and weekend indicators.

The Transformer uses historical daily power profiles, one-day lagged power values, calendar features, and daily load statistics to predict the following daily profile.

## Reported Results

The following values are transcribed from Tables I and II of the final report. Lower values indicate smaller errors. These are the original experimental results; they have not been reproduced as part of preparing this README.

| Model / experiment | Features | RMSE | MAE | sMAPE |
| --- | --- | ---: | ---: | ---: |
| LSTM V1, 2 layers | Basic temporal | 0.42 | 0.32 | 56.30% |
| LSTM, 3 layers | Temporal + weekly cyclic | 0.32 | 0.23 | 44.69% |
| **LSTM V2, 3 layers** | **Temporal + weekly cyclic + holidays** | **0.31** | **0.21** | **35.84%** |
| LSTM V3, 3 layers | V2 features + daily temperature | 0.54 | 0.39 | 58.77% |
| LSTM V4, 3 layers | V2 features + hourly temperature | 0.34 | 0.24 | 43.77% |
| XGBoost | Basic temporal | 0.96 | 0.76 | 106.14% |
| XGBoost | Lag and temporal features, including a 1-hour lag | 0.44 | 0.31 | 44.67% |
| XGBoost | Same lag feature set without the 1-hour lag | 0.75 | 0.57 | 76.80% |

The Transformer section separately reports **RMSE 0.633880162**, **MAE 0.46555078**, and **MAPE 4,340,506.5**. MAPE and sMAPE are different metrics and should not be compared directly. MAPE is sensitive to near-zero actual loads, which is relevant when interpreting this unusually large value. The Transformer also uses a different input representation and prediction setup, so its scores should be interpreted in that context.

### Findings

- The three-layer LSTM with holiday features performed best among the reported experiments.
- Adding more features did not consistently improve predictions: daily temperature degraded LSTM performance, while hourly temperature recovered some accuracy without surpassing V2.
- XGBoost benefited substantially from explicit lag features, especially the 1-hour lag.
- The Transformer captured broad trends and cycles but tended to smooth sharp peaks and troughs.
- Forecasting with lag features requires care when recent true load values are unavailable. Weather alignment and feature selection also affect performance.

## Repository Contents

```text
.
├── README.md
└── FinalProject/
    ├── CS412_Final_Report.pdf
    ├── LSTM_XGBoost_with_time_features_only.ipynb
    ├── LSTM_XGBoost_V2_best_added_holiday.ipynb
    ├── LSTM_XGBoost_V3_added_temperature.ipynb
    ├── LSTM_XGBoost_V4_added_hourly.ipynb
    └── Hanjun_transformer-final.zip
```

The Transformer archive contains:

```text
transformer/
├── data/
│   ├── train_data.xlsx
│   ├── validation_data.xlsx
│   └── check_data_to_students.xlsx
├── train.py
├── draw.py
└── transformer/
    ├── transformer_lag.pth
    ├── check_predictions_trans_lag.csv
    ├── transformer_validation_best_day_detail.png
    └── transformer_validation_overall_lag.png
```

## Getting Started

### 1. Clone and extract the data and Transformer code

```bash
git clone https://github.com/sylviaaacys/time-series-prediction.git
cd time-series-prediction
python -m zipfile -e FinalProject/Hanjun_transformer-final.zip FinalProject/
```

### 2. Prepare a Python environment

The source files use the following packages. The project does not include a pinned dependency file, so this is a starting environment rather than a verified reproduction environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab pandas numpy matplotlib seaborn scikit-learn openpyxl tensorflow torch xgboost openmeteo-requests requests-cache retry-requests
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### 3. Explore the notebooks

```bash
jupyter lab
```

Start with `FinalProject/LSTM_XGBoost_V2_best_added_holiday.ipynb` for the best reported LSTM configuration, or use the other notebooks to inspect feature experiments.

Before running cells:

- Update the original `/content/train_data.xlsx` and `/content/validation_data.xlsx` paths to the extracted Excel files under `FinalProject/transformer/data/`. These paths reflect the original notebook environment.
- Review weather API cells and time alignment before retrieving external data.
- The time-features notebook also references `/content/weather_data_processed.csv`, which is **not included** in the archive. Supply that file or adapt the relevant cells.
- Review saved-model paths and cell order. The notebooks retain the original exploratory workflow and may require adjustments for a clean run.

### 4. Train the Transformer

In the extracted `FinalProject/transformer/train.py`, change `base_path` inside `transformer()` from its original absolute path to the absolute path of your extracted `FinalProject/transformer/` directory.

```bash
python FinalProject/transformer/train.py
```

The script selects CUDA when available and otherwise uses the CPU. It trains the model, saves the checkpoint with the best validation RMSE, creates validation plots, and exports checking-set predictions under `FinalProject/transformer/transformer/`. Running it can overwrite the bundled outputs in that directory.

## Authors

**Hanjun Luo · Sylvia Chung Yan Shan · Thiwarin Sudjaipraparat**  
Electronics and Computer Engineering, Zhejiang University

## References

- [CS412 Final Project: Load Data Prediction Model](FinalProject/CS412_Final_Report.pdf) - primary source for the study and reported results.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) - weather data source cited in the report.
- [Public Holidays in China](https://www.timeanddate.com/holidays/china/) - holiday reference cited in the report.
