## **Sales Forecasting using LSTM and Power BI Visualization**

### **Project Overview**
This project aims to build a machine learning model using a **Long Short-Term Memory (LSTM)** network to forecast sales for a retail store. The model was developed using **Python**, and the results were visualized using **Power BI**. The key goal is to accurately predict future sales and display insights via dashboard.

### **Motivation**
Sales forecasting is a critical task for retail stores to manage inventory, marketing, and operations effectively. The **LSTM** network is well-suited for time series forecasting, as it can capture long-term dependencies and trends in historical sales data. In this project, we focus on building an LSTM model to predict future sales and visualize the actual vs. predicted sales using **Power BI**.
### **Context**
This is one of my first projects using **Machine Learning** in order to predict data!



---

### **Dataset**
- **Dataset**: Rossmann Store Sales
- **Source**: [Kaggle Rossmann Store Sales Dataset](https://www.kaggle.com/c/rossmann-store-sales/data)
- **Features**:
  - **Date**: The date of sales.
  - **Sales**: The actual sales for a given date (target variable).
  - **Promo**: Whether the store had a promotion that day.
  - **StateHoliday**: Whether it was a state holiday.
  - **SchoolHoliday**: Whether it was a school holiday.
  - **DayOfWeek**: The day of the week for the sales.
  - **Month**: Month of the year.
  - **Lagged Sales**: Previous day’s sales.
  - **Rolling Averages**: 7-day and 30-day rolling averages for sales.

---

### **Model Description**
The model was built using **Long Short-Term Memory (LSTM)** networks, which are a type of recurrent neural network (RNN) that can capture dependencies in time series data. The model was trained to predict future sales based on historical sales and several features, including promotional information and holidays.

**Key Model Details**:
- **Time Steps**: 90 days (90-day historical window to predict the next day’s sales).
- **Model Architecture**:
  - Two layers of **Bidirectional LSTMs** with dropout to prevent overfitting.
  - The model was compiled with the **Adam optimizer** and trained over 100 epochs.
- **Batch Size**: 32 (after tuning).

---

### **Results**

The model was evaluated using several metrics:

- **Mean Absolute Error (MAE)**: 651.08
- **Mean Squared Error (MSE)**: 1,311,146.32
- **Root Mean Squared Error (RMSE)**: 1,145.53
- **R-squared (R²)**: 0.7124

The LSTM model captured **71.24% of the variance** in the sales data. While the RMSE and MAE indicate that the model has some error, the results show that the LSTM is performing well for the sales forecasting task.

---

### **Power BI Visualization**
To better communicate the results, the predicted vs. actual sales were visualized using **Power BI**. The Power BI dashboard provides an interactive and intuitive way to explore the forecasting results, including:

- **Line Chart**: A comparison between actual and predicted sales over time.
- **KPIs**: Visual representations of key performance indicators like MAE, RMSE, and R².


The Power BI `.pbix` file can be downloaded and viewed to explore the visualizations and insights further.

---
