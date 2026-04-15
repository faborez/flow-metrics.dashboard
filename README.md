# 📊 Flow Metrics Dashboard

A specialized analytics tool built for software delivery teams to visualize flow metrics and improve predictability. This dashboard is designed to process data exports from the **Status Time Reports** (Pro or Free) Jira plugins, converting raw status entry dates into actionable insights.

## 🚀 Live Access
You can run your own analysis immediately using the cloud-hosted version of this dashboard:
**[Launch Flow Metrics Dashboard on Streamlit](https://flow-metrics-dashboard-status-time-reports.streamlit.app/)**

---

## 📖 Overview
Modern agile teams rely on flow metrics to understand how work moves through their system. This dashboard automates the calculation of the "Big Four" flow metrics and provides advanced statistical forecasting using Monte Carlo simulations.

### Key Metrics Included:
* **Cycle Time:** Track the time taken from "Start" to "Done" for every work item.
* **Work Item Age:** Identify items currently in progress that are exceeding historical norms.
* **Work In Progress (WIP):** Monitor the total load on the team to prevent bottlenecks.
* **Throughput:** Measure the rate of delivery (items completed per period).
* **Flow Efficiency:** Visualize the ratio of active work time versus waiting time.

---

## 🛠️ Features
* **Monte Carlo Forecasting:** Predict "How Many" items can be done by a date or "When" a specific backlog will be finished.
* **Interactive Filtering:** Dynamically filter by Work Type, Team, Labels, or any custom Jira field.
* **Stability Check:** Built-in statistical analysis to determine if your historical data is consistent enough for reliable forecasting.
* **Color-Blind Friendly Mode:** Toggleable color palettes designed for accessibility.
* **Data Privacy:** All processing happens locally in your browser; no data is ever uploaded to a server.

---

## 💻 Local Installation & Setup

If you prefer to run the dashboard locally, follow these steps:

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed.

### 2. Clone and Install
```bash
# Clone the repository
git clone https://github.com/your-username/flow-metrics-dashboard.git
cd flow-metrics-dashboard

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run flow_metrics_dashboard.py
```

---

## 📋 Data Export Instructions
To get the most out of the dashboard, export your data from Jira using the **Status Time Reports** plugin with these settings:

1.  **Select 'Show entry dates'**: This is essential for calculating flow.
2.  **Status Column Order**: Arrange your columns from left-to-right to match your workflow (e.g., *To Do → In Progress → Done*).
3.  **Required Fields**: Ensure 'Key' and 'Work type' are included in the export.
4.  **Security**: Avoid exporting sensitive PII (Personally Identifiable Information).

---

## 📄 Requirements
The project relies on several key data science libraries including:
* `streamlit`: For the web interface.
* `pandas` & `numpy`: For data processing and math.
* `plotly`: For interactive charting.
* `scikit-learn`: For trend line analysis.

---
*Developed to help teams make data-driven decisions through the power of flow.*
