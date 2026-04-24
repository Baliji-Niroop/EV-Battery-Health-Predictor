# 📘 Complete Guide: EV Battery Health Project

Hey! Welcome to the detailed walkthrough of my EV Battery Health & Remaining Useful Life (RUL) estimation project. If you've just cloned the repo and have zero idea what's going on, this guide is for you. I'll walk you through exactly what this project does, how the data flows, and how the code actually works under the hood.

---

## Start Here (30 seconds)

If you just want to run the project:

1. Download dataset:
   https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

2. Put .mat files inside:
   data/raw/

3. Run:
   python run_pipeline.py

4. Launch dashboard:
   streamlit run dashboard/app.py

---

## 1. Introduction

### What is this project?
This project is an end-to-end Machine Learning pipeline that predicts how long an Electric Vehicle (EV) battery will last before it needs to be replaced. 

### Why does battery health matter?
Batteries don't last forever. Every time you charge and discharge a lithium-ion battery, it degrades a tiny bit. If an EV battery fails unexpectedly, it can leave someone stranded or even be a safety hazard. Plus, replacing an EV battery is incredibly expensive.

### What problem are we solving?
We are trying to predict the future. Specifically, we want to know two things at any given time:
1. **State of Health (SoH):** "How healthy is the battery right now compared to when it was brand new?"
2. **Remaining Useful Life (RUL):** "How many more charge cycles can this battery survive before it's basically dead?"

By answering these questions, we can schedule maintenance *before* the battery completely fails.

---

## 2. Big Picture (High Level)

Before we dive into the code, here is the basic flow of the project:

1. **Raw Data:** We start with raw sensor readings (voltage, current, temp) stored in messy files.
2. **Preprocessing:** We clean up the mess and turn it into a neat, readable CSV spreadsheet.
3. **Features:** We calculate new, helpful metrics (like how fast the capacity is dropping).
4. **Model:** We train an AI (XGBoost) to learn the patterns of a dying battery.
5. **RUL:** We use the AI to guess exactly when the battery will cross the "danger zone."
6. **Dashboard:** We display everything on a clean web app so a human can actually read it.

---

## 3. Dataset Explanation

### The NASA Battery Dataset
I used a public dataset from NASA. NASA ran lithium-ion batteries through thousands of charge and discharge cycles until they essentially died, recording all the physical sensor data along the way.

### What are `.mat` files?
The data comes in `.mat` files. These are MATLAB files. They are incredibly annoying to work with in Python because the data is nested in deep, weird dictionary-like structures.

### What's inside the data?
For every single charge cycle, NASA recorded:
- **Voltage:** The electrical pressure.
- **Current:** The flow of electricity.
- **Temperature:** How hot the battery got.
- **Capacity:** The total amount of energy the battery could hold during that cycle.

### Why is the data messy?
Sensors aren't perfect. Sometimes there are gaps in the data, the structures vary slightly, and the arrays of numbers are buried five layers deep inside MATLAB objects. Our first big task was just digging that data out safely.

---

## 4. Step-by-Step Pipeline (Core Section)

Here is a breakdown of every script in the `src/` folder and what it actually does.

### 4.1 `preprocess.py`
This is the heavy lifter. It opens those nasty `.mat` files and rips out the useful data.
- **How it works:** It uses a library called `scipy.io` to read the MATLAB files, hunts down the nested array that holds the "cycles," and extracts them one by one.
- **What it computes:** For every single cycle, it calculates the average voltage (`voltage_mean`), average current (`current_mean`), average temperature (`temp_mean`), and grabs the final `capacity`.
- **Calculating SoH:** It figures out the battery's absolute maximum capacity (usually cycle 1) and defines that as 100%. Then, for every subsequent cycle, it calculates the **State of Health (SoH)** as `Current Capacity / Max Capacity`.
- **Challenges:** Writing the logic to blindly navigate MATLAB structs without crashing if a field was missing or misnamed was surprisingly tricky!

### 4.2 `features.py`
Raw sensor data usually isn't enough for an ML model to learn effectively. We need to give the AI some "context."
- **Why we need it:** Just knowing the temperature *today* isn't as helpful as knowing the temperature trend over the last *week*.
- **What it adds:**
  - `capacity_fade_rate`: How much capacity was lost between yesterday and today? (If this accelerates, the battery is dying fast).
  - `rolling averages`: It smooths out the data by taking the average of the last 5 cycles, hiding random sensor noise.

### 4.3 `anomaly_model.py`
This is where the actual Machine Learning happens.
- **What it predicts:** We train the model to look at the features (voltage, temp, fade rate) and guess the `SoH`.
- **How it trains:** We split our data. 80% is used to teach the model, and 20% is hidden away to test it later.
- **Metrics (MAE & RMSE):** 
  - MAE (Mean Absolute Error) is simply "on average, how many percentage points was the model's guess off by?"
  - RMSE (Root Mean Squared Error) is similar, but it heavily penalizes the model if it makes really huge, crazy errors.

#### Why this approach?
I used XGBoost instead of deep learning because:
- Dataset is relatively small
- Tabular data works well with tree models
- Faster to train and debug

### 4.4 `rul_estimator.py`
Now that we have a trained model, we use it to predict the future.
- **What is RUL?** Remaining Useful Life.
- **The 80% Rule:** In the EV industry, a battery is considered "dead" (or in need of replacement) when its SoH drops below 80%. 
- **How we calculate it:** We ask the XGBoost model to predict the SoH for all future cycles. We find the exact cycle where the prediction drops below 80. Then, we subtract the *current* cycle. (e.g., If it dies on cycle 150, and we are on cycle 100, the RUL is 50).

#### Example (What actually happens)
For one battery:

- Cycle 1 → SoH = 100%
- Cycle 50 → SoH = 92%
- Cycle 100 → SoH = 81%
- Cycle 120 → SoH = 79% → considered failure

So:
RUL = 120 - current_cycle

### 4.5 `dashboard/app.py`
A machine learning model isn't very useful if it just prints numbers to a terminal. 
- **What the user sees:** A clean, interactive web page built with Streamlit.
- **The interface:** There's a sidebar to select different batteries from the dataset. It shows the current health, temperature, and a big warning banner if the battery is in bad shape.
- **The graphs:** It plots the actual health of the battery against what our AI *predicted* the health would be, making it easy to visually verify if our model is hallucinating or not.

---

## 5. Data Flow Diagram

Here is a visual map of how data moves through the project:

```text
[NASA .mat Files] 
       │
       ▼
(preprocess.py)  ──>  extracts sensors & capacity
       │
       ▼
[cleaned_data.csv]
       │
       ▼
(features.py)    ──>  adds rolling averages & fade rates
       │
       ▼
[features.csv]
       │
       ▼
(anomaly_model.py) ──> trains the AI
       │
       ▼
[xgb_model.pkl]
       │
       ▼
(rul_estimator.py) ──> calculates cycles until 80% health
       │
       ▼
(dashboard/app.py) ──> displays graphs to the user
```

---

## 6. How to Run the Project

Running this is designed to be as simple as possible. You only need two commands.

1. **Run the entire ML pipeline (Data -> Training -> Predictions):**
```bash
python run_pipeline.py
```

2. **Open the web dashboard:**
```bash
streamlit run dashboard/app.py
```

---

## 7. Example Output

When you run the pipeline, the terminal will print out something like this:

```text
[INFO] Battery B0005:
       Current cycle: 168
       Current SoH: 72.50%
       Predicted failure cycle: 122
       RUL (cycles remaining): 0
```
**What this means:** Battery B0005 is currently on cycle 168. Its health is 72.5%. Our model predicted it *should* have failed around cycle 122. Because it's already past cycle 122, the Remaining Useful Life is 0. It's officially dead!

### What the output looks like

- A graph showing actual vs predicted SoH
- A number showing remaining useful life (RUL)
- A dashboard with battery selection and trends

---

## 8. Common Errors and Fixes

- **"Error: No .mat or .zip files found"**
  - **Fix:** You forgot to download the dataset! Go to the NASA repository, download the battery data, and drop the `.mat` files into the `data/raw/` folder.
- **"Error: cleaned_data.csv is missing!"**
  - **Fix:** You tried to run feature engineering before preprocessing. Just use `python run_pipeline.py` to run things in the correct order.
- **"Model not found"**
  - **Fix:** The XGBoost model hasn't been trained yet. Run `python src/anomaly_model.py` first.

---

## Limitations

- Uses historical data, not real-time streaming
- Assumes failure threshold at 80% SoH (simplification)
- Model does not account for external conditions like driving behavior

---

## 9. What I Learned

This project was a massive learning experience for me.
- **Real-world data is messy:** I spent way more time writing the `preprocess.py` script to parse MATLAB files than I did actually writing the AI model. Data cleaning is 80% of the job!
- **Simple models win:** I originally thought about using a crazy Deep Learning Neural Network, but XGBoost handled this tabular time-series data perfectly and trained in a fraction of the time.
- **Streamlit is awesome:** Building a frontend UI used to take weeks of JavaScript. Streamlit let me build a professional dashboard in pure Python in just a few hours.

---

## 10. Future Improvements

If I had more time to work on this, here is what I would add:
- **Better Time-Series Models:** I'd like to try an LSTM (Long Short-Term Memory) neural network, which is specifically designed to remember past events in time sequences.
- **Real-time Data:** Instead of reading static CSV files, it would be cool to simulate real-time sensor data streaming in over an MQTT or Kafka pipeline.
- **More Features:** I want to calculate internal resistance, as that is a huge physical indicator of battery death.

Thanks for reading! Feel free to explore the code.
