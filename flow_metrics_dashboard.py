import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import re
from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame
from plotly.graph_objs import Figure
import base64
import os

# Configuration
st.set_page_config(page_title="Flow Metrics Dashboard", layout="wide")

class Config:
    """Centralized configuration for the dashboard."""
    WORK_TYPE_COLORS = {
        'Epic': '#8B5CF6', 'Story': '#10B981', 'Task': '#3B82F6',
        'Bug': '#EF4444', 'Spike': '#F97316'
    }
    DEFAULT_COLOR = '#808080'
    WORK_TYPE_ORDER = ['Epic', 'Story', 'Task', 'Bug', 'Spike']
    DATE_RANGES = ["All time", "Last 30 days", "Last 60 days", "Last 90 days", "Custom"]
    
    PERCENTILES = [50, 70, 85, 95]
    PERCENTILE_COLORS = { 50: "red", 70: "orange", 85: "green", 95: "blue" }

    FORECAST_LIKELIHOODS = {
        95: 5,
        85: 15,
        70: 30,
        50: 50
    }
    THROUGHPUT_INTERVALS = ["Weekly", "Fortnightly", "Monthly"]
    FORECASTING_SIMULATIONS = 10000
    FORECAST_DATE_RANGES = ["Next 30 days", "Next 60 days", "Next 90 days", "Custom"]
    
    OPTIONAL_FILTERS = {
        "Team": "single", "Labels": "multi", "Components": "multi",
        "High Level Estimate-DPID": "multi", "RAG-DPID": "multi"
    }

class ChartConfig:
    """Centralized configuration for chart templates and layouts."""
    CYCLE_TIME_HOVER = (
        "%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br>"
        "<b>Completed:</b> %{customdata[2]}<br><b>Start:</b> %{customdata[3]}<br>"
        "<b>Cycle time:</b> %{customdata[4]} days<extra></extra>"
    )
    AGE_CHART_HOVER = (
        "%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br><b>Status:</b> %{customdata[2]}<br>"
        "<b>Start:</b> %{customdata[3]}<br><b>Age:</b> %{customdata[4]} days<extra></extra>"
    )
    WIP_CHART_HOVER = "<b>Date:</b> %{x|%d/%m/%Y}<br><b>WIP count:</b> %{y}<extra></extra>"
    THROUGHPUT_CHART_HOVER = (
        "<b>Throughput Chart</b><br>Period = %{customdata[0]}<br>"
        "Throughput = %{y} items<br>%{customdata[1]}<extra></extra>"
    )
    TREND_LINE_HOVER = "<b>Trend Line</b><br>Date = %{x|%d/%m/%Y}<br>Trend value = %{y:.1f}<extra></extra>"


class StatusManager:
    """Handles status column extraction and validation."""
    @staticmethod
    def extract_status_columns(df: DataFrame) -> Dict[str, str]:
        """Extracts columns that represent a status transition."""
        return {col.replace("'->", "").strip(): col for col in df.columns if col.startswith("'->")}

    @staticmethod
    def validate_status_order(df: DataFrame, start_col: Optional[str], completed_col: Optional[str]) -> Tuple[bool, str]:
        """Validates that the start status comes before the completed status in the DataFrame."""
        if not start_col or not completed_col:
            return False, "Both start and completed statuses must be selected."
        if start_col == completed_col:
            return False, "Start and completed status cannot be the same."
        try:
            columns = list(df.columns)
            if columns.index(start_col) >= columns.index(completed_col):
                return False, "Start status must be earlier in the workflow than completed status."
            return True, ""
        except ValueError:
            return False, "Invalid status columns selected."


class DataProcessor:
    """Handles loading and processing of JIRA export data."""
    @staticmethod
    @st.cache_data
    def load_data(uploaded_file) -> Optional[DataFrame]:
        """Loads data from the uploaded CSV file."""
        try:
            df = pd.read_csv(uploaded_file, keep_default_na=False, encoding='utf-8')
            df = df.dropna(how='all')
            if not {'Key', 'Issue Type'}.issubset(df.columns):
                st.error("Invalid file format: CSV must include 'Key' and 'Issue Type' columns.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def clean_data(df: DataFrame) -> DataFrame:
        """Cleans the raw DataFrame, handling duplicates and renaming columns."""
        df_clean = df.rename(columns={'Issue Type': 'Work type'})
        
        if df_clean.duplicated(subset=['Key']).any():
            duplicates = df_clean[df_clean.duplicated(subset=['Key'], keep=False)]
            st.warning(
                f"Found and removed {len(duplicates.drop_duplicates(subset=['Key']))} duplicate work item key(s). "
                f"The first occurrence of each was kept. Example duplicate key: {duplicates['Key'].iloc[0]}"
            )
        
        df_clean = df_clean.drop_duplicates(subset=['Key'], keep='first').copy()
        if 'Work type' in df_clean.columns:
            df_clean.loc[:, 'Work type'] = df_clean['Work type'].str.strip()
        return df_clean

    @staticmethod
    def process_dates(df: DataFrame, start_col: Optional[str], completed_col: Optional[str]) -> Optional[DataFrame]:
        """Processes date columns and calculates cycle time."""
        try:
            processed_df = df.copy()
            processed_df['Start date'] = pd.to_datetime(processed_df[start_col].apply(DataProcessor._extract_earliest_date) if start_col else pd.NaT, errors='coerce')
            processed_df['Completed date'] = pd.to_datetime(processed_df[completed_col].apply(DataProcessor._extract_earliest_date) if completed_col else pd.NaT, errors='coerce')
            return DataProcessor._calculate_cycle_time(processed_df)
        except Exception as e:
            st.error(f"Error processing dates: {str(e)}")
            return None

    @staticmethod
    def _extract_earliest_date(date_str: Union[str, float]) -> Optional[str]:
        """Extracts the earliest date from a string that may contain multiple dates."""
        if pd.isna(date_str) or str(date_str).strip() in ['-', '', 'nan']:
            return None
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', str(date_str))
        return min(dates) if dates else None

    @staticmethod
    def _calculate_cycle_time(df: DataFrame) -> DataFrame:
        """Calculates cycle time for items with start and completed dates."""
        df = df[df['Work type'].notna() & (df['Work type'] != '')].copy()
        df['Cycle time'] = (df['Completed date'] - df['Start date']).dt.days + 1
        invalid_cycle = (df['Cycle time'].notna()) & (df['Cycle time'] < 1)
        if invalid_cycle.any():
            st.warning(f"Removed {invalid_cycle.sum()} items with invalid cycle times (< 1 day).")
            df = df[~invalid_cycle]
        return df


class ChartGenerator:
    """Generates Plotly charts for the dashboard."""
    @staticmethod
    def create_cycle_time_chart(df: DataFrame, percentile_settings: Dict[str, bool]) -> Optional[Figure]:
        """Creates the cycle time scatterplot."""
        completed_df = df.dropna(subset=['Start date', 'Completed date', 'Cycle time'])
        if completed_df.empty:
            return None

        chart_df = ChartGenerator._prepare_chart_data(completed_df, ['Key', 'Work type', 'Completed date', 'Start date', 'Cycle time'])
        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scatter(
                x=df_type['Completed date'], y=df_type['Cycle time'], mode='markers', name=work_type,
                marker=dict(color=Config.WORK_TYPE_COLORS.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7),
                customdata=df_type[['Key', 'Work type', 'Completed_date_formatted', 'Start_date_formatted', 'Cycle_time_formatted']],
                hovertemplate=ChartConfig.CYCLE_TIME_HOVER
            ))
        
        fig.update_layout(title="Cycle Time Scatterplot", xaxis_title="Completed Date", yaxis_title="Cycle Time (Days)", height=600, legend_title="Work Type")
        
        for p, color in Config.PERCENTILE_COLORS.items():
            if percentile_settings.get(f"show_{p}th", True):
                y_value = np.percentile(chart_df["Cycle time"], p)
                hover_text = f"<b>{p}th Percentile</b><br>Value: {int(y_value)} days<br><i>{p}% of items finish in this time or less.</i>"
                ChartGenerator._add_hoverable_line(fig, y_value, chart_df["Completed date"], hover_text, color, f"{p}th: {int(y_value)}d")
        
        return fig

    @staticmethod
    def create_cycle_time_histogram(df: DataFrame, percentile_settings: Dict[str, bool]) -> Optional[Figure]:
        """Creates a histogram of cycle time distribution."""
        completed_df = df.dropna(subset=['Cycle time'])
        if completed_df.empty:
            return None

        fig = px.histogram(
            completed_df, x="Cycle time",
            title="Cycle Time Distribution",
            labels={'Cycle time': 'Cycle Time (Days)', 'count': 'Number of Work Items'},
            color_discrete_sequence=['#3B82F6']
        )
        fig.update_layout(bargap=0.1, yaxis_title="Number of Work Items")

        for p, color in Config.PERCENTILE_COLORS.items():
            if percentile_settings.get(f"show_{p}th", True):
                percentile_val = np.percentile(completed_df['Cycle time'], p)
                fig.add_vline(
                    x=percentile_val, line_dash="dash", line_color=color,
                    annotation_text=f"{p}th: {int(percentile_val)}d",
                    annotation_position="top right"
                )
        return fig
    
    @staticmethod
    def create_time_in_status_chart(df: DataFrame, status_cols: List[str]) -> Tuple[Optional[Figure], Optional[DataFrame]]:
        """Calculates and creates a bar chart of the average time spent in each status."""
        
        if len(status_cols) < 2:
            return None, None

        all_durations = []
        for i in range(len(status_cols) - 1):
            current_col = status_cols[i]
            next_col = status_cols[i+1]
            
            temp_df = df[[current_col, next_col]].copy()
            
            temp_df['current_date'] = pd.to_datetime(temp_df[current_col].apply(DataProcessor._extract_earliest_date), errors='coerce')
            temp_df['next_date'] = pd.to_datetime(temp_df[next_col].apply(DataProcessor._extract_earliest_date), errors='coerce')
            
            temp_df.dropna(subset=['current_date', 'next_date'], inplace=True)
            
            if temp_df.empty:
                continue
                
            temp_df['duration'] = (temp_df['next_date'] - temp_df['current_date']).dt.days
            
            valid_durations = temp_df[temp_df['duration'] >= 0]
            
            if not valid_durations.empty:
                avg_duration = np.ceil(valid_durations['duration'].mean())
                status_name = current_col.replace("'->", "").strip()
                all_durations.append({'Status': status_name, 'Average Time (Days)': avg_duration})

        if not all_durations:
            return None, None

        chart_df = pd.DataFrame(all_durations)
        
        fig = px.bar(
            chart_df,
            x='Status',
            y='Average Time (Days)',
            title='Average Time in Each Status',
            text='Average Time (Days)'
        )
        fig.update_traces(texttemplate='%{text:.0f}d', textposition='outside', textfont_size=14)
        fig.update_layout(
            yaxis_title="Average Time (Days)", 
            font=dict(size=14),
            yaxis_range=[0, chart_df['Average Time (Days)'].max() * 1.15]
        )
        return fig, chart_df

    @staticmethod
    def create_work_item_age_chart(df: DataFrame, axis_start_col: str, axis_done_col: str, cycle_time_percentiles: Dict[str, int], percentile_settings: Dict[str, bool]) -> Optional[Figure]:
        """Creates the work item age chart."""
        df_in_progress = df[df['Completed date'].isna()].copy()
        if df_in_progress.empty:
            return None

        age_data = []
        for _, row in df_in_progress.iterrows():
            start_date_str = DataProcessor._extract_earliest_date(row[axis_start_col])
            if start_date_str:
                age_data.append({
                    'Key': row['Key'], 'Work type': row['Work type'], 'Status': row['Status'],
                    'Age': (datetime.now() - pd.to_datetime(start_date_str)).days + 1,
                    'Start date': pd.to_datetime(start_date_str)
                })
        
        if not age_data:
            return None
        
        try:
            all_status_cols = list(df.columns)
            start_idx = all_status_cols.index(axis_start_col)
            end_idx = all_status_cols.index(axis_done_col)
            status_order = [s.replace("'->", "").strip() for s in all_status_cols[start_idx : end_idx + 1]]
        except ValueError:
            return None
        
        age_df = pd.DataFrame(age_data)
        chart_df = ChartGenerator._prepare_chart_data(age_df, ['Key', 'Work type', 'Status', 'Age', 'Start date'])
        chart_df['Status'] = pd.Categorical(chart_df['Status'], categories=status_order, ordered=True)
        chart_df.dropna(subset=['Status'], inplace=True)

        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scatter(
                x=df_type['Status'], y=df_type['Age'], mode='markers', name=work_type,
                marker=dict(color=Config.WORK_TYPE_COLORS.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7),
                customdata=df_type[['Key', 'Work type', 'Status', 'Start_date_formatted', 'Age_formatted']],
                hovertemplate=ChartConfig.AGE_CHART_HOVER
            ))
        
        num_statuses = len(status_order)
        domain_end = min(1.0, num_statuses / 6.0) if num_statuses > 0 else 1.0

        fig.update_layout(
            title="Work Item Age Chart", xaxis_title="Status", yaxis_title="Age (Calendar Days)", 
            height=600, legend_title="Work Type",
            xaxis=dict(
                domain=[0, domain_end], 
                categoryorder='array', 
                categoryarray=status_order, 
                range=[-0.5, num_statuses - 0.5],
                showgrid=True,
                gridcolor='LightGrey',
                gridwidth=1
            ),
            legend=dict(x=domain_end * 1.02, xanchor='left')
        )
        
        for p_int_str, p_val in cycle_time_percentiles.items():
            if p_int_str.startswith('p'):
                p = int(p_int_str[1:])
                if percentile_settings.get(f"show_{p}th", True):
                    hover_text = f"<b>{p}th Percentile (from Cycle Time)</b><br>Value: {p_val} days<br><i>Items above this line are aging longer than {p}% of past items.</i>"
                    ChartGenerator._add_hoverable_line(fig, p_val, status_order, hover_text, Config.PERCENTILE_COLORS.get(p), f"{p}th: {p_val}d")
        return fig

    @staticmethod
    def _add_hoverable_line(fig, y_value, x_data, hover_text, color, annotation_text):
        """Helper to add a horizontal line with a hover-sensitive area."""
        fig.add_hline(y=y_value, line_dash="dash", line_color=color, line_width=1.5, annotation_text=annotation_text, annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=list(x_data), y=[y_value] * len(x_data),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=20),
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    @staticmethod
    def create_wip_chart(df: DataFrame) -> Optional[Figure]:
        """Creates the WIP (Work In Progress) run chart."""
        wip_df = df.dropna(subset=['Start date'])
        if wip_df.empty:
            return None

        starts = wip_df[['Start date']].rename(columns={'Start date': 'Date'})
        starts['Change'] = 1
        
        ends = wip_df.dropna(subset=['Completed date'])[['Completed date']].rename(columns={'Completed date': 'Date'})
        ends['Change'] = -1

        wip_events = pd.concat([starts, ends]).sort_values(by='Date')
        wip_over_time = wip_events.groupby(pd.Grouper(key='Date', freq='D')).sum().cumsum().reset_index()
        wip_over_time.rename(columns={'Change': 'WIP'}, inplace=True)
        
        fig = px.line(wip_over_time, x="Date", y="WIP", title="WIP (Work In Progress) Run Chart")
        fig.update_traces(hovertemplate=ChartConfig.WIP_CHART_HOVER)
        
        ChartGenerator._add_trend_line(fig, wip_over_time)
        return fig

    @staticmethod
    def create_throughput_chart(df: DataFrame, interval: str, throughput_status_col: str) -> Optional[Figure]:
        """Creates the throughput bar chart."""
        if not throughput_status_col:
            return None

        throughput_df = df.copy()
        throughput_df['Throughput Date'] = pd.to_datetime(throughput_df[throughput_status_col].apply(DataProcessor._extract_earliest_date), errors='coerce')
        throughput_df.dropna(subset=['Throughput Date'], inplace=True)

        if throughput_df.empty:
            return None
            
        period_map = {"Weekly": 'W-MON', "Fortnightly": '2W-MON', "Monthly": 'MS'}
        grouper = pd.Grouper(key='Throughput Date', freq=period_map.get(interval, 'W-MON'))
        
        agg_df = throughput_df.groupby(grouper).agg(
            Throughput=('Key', 'count'),
            Details=('Work type', lambda s: '<br>'.join(f"{wt}: {count}" for wt, count in s.value_counts().items()))
        ).reset_index().rename(columns={'Throughput Date': 'Period'})
        
        agg_df['Details'] = "<b>Breakdown:</b><br>" + agg_df['Details']
        agg_df['Period_formatted'] = agg_df['Period'].dt.strftime('%d/%m/%Y')
        
        title_interval = interval.replace("ly", "")
        fig = px.bar(agg_df, x="Period", y="Throughput", title=f"Throughput per {title_interval}", text="Throughput")
        fig.update_traces(
            textposition='outside',
            hovertemplate=ChartConfig.THROUGHPUT_CHART_HOVER,
            customdata=agg_df[['Period_formatted', 'Details']].values
        )
        if not agg_df.empty:
            fig.update_layout(yaxis_range=[0, agg_df['Throughput'].max() * 1.15])
            
        return fig

    @staticmethod
    @st.cache_data
    def _run_how_many_simulation(weekly_throughput: pd.Series, normalized_weights: np.ndarray, forecast_days: int) -> np.ndarray:
        """Cached function to run the core 'how many' simulation."""
        num_weeks = forecast_days / 7.0
        num_simulation_weeks = int(np.ceil(num_weeks))
        simulations = np.random.choice(weekly_throughput, (Config.FORECASTING_SIMULATIONS, num_simulation_weeks), replace=True, p=normalized_weights)
        
        forecast_counts = simulations.sum(axis=1)

        if num_weeks % 1 != 0:
            fractional_week_multiplier = num_weeks % 1
            last_week_simulation = np.random.choice(weekly_throughput, Config.FORECASTING_SIMULATIONS, replace=True, p=normalized_weights)
            forecast_counts -= ((1 - fractional_week_multiplier) * last_week_simulation).astype(int)
        
        return forecast_counts

    @staticmethod
    def create_how_many_forecast_chart(df: DataFrame, forecast_days: int, throughput_status_col: str) -> Optional[Figure]:
        """Prepares data and calls the cached simulation to create a 'How Many' forecast chart."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        
        if weekly_throughput is None:
            return None

        with st.spinner(f"Running {Config.FORECASTING_SIMULATIONS} weighted simulations..."):
            forecast_counts = ChartGenerator._run_how_many_simulation(weekly_throughput, normalized_weights, forecast_days)

        counts, bin_edges = np.histogram(forecast_counts, bins=30, range=(forecast_counts.min(), forecast_counts.max()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = go.Figure(data=[go.Bar(x=bin_centers, y=counts, name='Simulations')])
        fig.update_layout(
            title=f"Forecast: How Many Items in the Next {forecast_days} Days?",
            xaxis_title="Number of Items Completed", yaxis_title="Frequency",
            bargap=0.1, yaxis_range=[0, counts.max() * 1.20]
        )

        summary_text = f"**Forecast Summary (for next {forecast_days} days):**"
        for likelihood, percentile in sorted(Config.FORECAST_LIKELIHOODS.items(), reverse=True):
            value = np.percentile(forecast_counts, percentile)
            color = Config.PERCENTILE_COLORS.get(likelihood)
            fig.add_vline(
                x=value, line_dash="dash", line_color=color,
                annotation_text=f"{likelihood}%: {int(value)}",
                annotation_position="top left"
            )
            summary_text += f"\n- There is a **{likelihood}% chance** to complete **{int(value)} or more** items."
        
        st.markdown(summary_text)
        return fig

    @staticmethod
    @st.cache_data
    def _run_when_simulation(items_to_complete: int, _full_throughput_dataset: pd.Series, _weights: np.ndarray, _first_week_sample_dataset: Optional[pd.Series] = None) -> List[int]:
        """A reusable, cached helper to run the 'when' simulation."""
        full_throughput_dataset = _full_throughput_dataset.to_numpy()
        weights = _weights
        first_week_sample_dataset = _first_week_sample_dataset.to_numpy() if _first_week_sample_dataset is not None else full_throughput_dataset

        first_week_weights = None
        if len(first_week_sample_dataset) < len(full_throughput_dataset):
            # This logic is complex to re-align, simpler to just use uniform for the filtered set
            first_week_weights = None 
        else:
            first_week_weights = weights

        completion_weeks_data = []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            weeks_elapsed = 0
            items_done = 0
            timeout = max(300, (items_to_complete / full_throughput_dataset.mean()) * 10)

            items_done += np.random.choice(first_week_sample_dataset, p=first_week_weights)
            weeks_elapsed += 1

            while items_done < items_to_complete:
                if weeks_elapsed > timeout:
                    weeks_elapsed = -1 
                    break
                items_done += np.random.choice(full_throughput_dataset, p=weights)
                weeks_elapsed += 1
            
            if weeks_elapsed != -1:
                completion_weeks_data.append(weeks_elapsed)
        
        return completion_weeks_data

    @staticmethod
    def _get_recent_weekly_throughput(df: DataFrame, status_col: str) -> Tuple[Optional[pd.Series], Optional[np.ndarray]]:
        """Gets recent weekly throughput and calculates sampling weights."""
        if not status_col:
            return None, None
            
        forecast_df = df.copy()
        forecast_df['Forecast Completion Date'] = pd.to_datetime(forecast_df[status_col].apply(DataProcessor._extract_earliest_date), errors='coerce')
        completed_df = forecast_df.dropna(subset=['Forecast Completion Date'])

        if len(completed_df) < 2:
            return None, None

        last_completion_date = completed_df['Forecast Completion Date'].max()
        start_of_period = last_completion_date - pd.DateOffset(weeks=25)
        recent_completed_df = completed_df[completed_df['Forecast Completion Date'] > start_of_period]

        if recent_completed_df.empty:
            st.warning("Not enough recent data for forecasting. Need completed items from the last 25 weeks.")
            return None, None
            
        weekly_throughput = recent_completed_df.groupby(pd.Grouper(key='Forecast Completion Date', freq='W-MON')).size()
        
        num_weeks_of_data = len(weekly_throughput)
        if num_weeks_of_data < 2:
            st.warning("Not enough weekly throughput samples in the last 25 weeks to forecast.")
            return None, None
        
        if num_weeks_of_data < 7:
            st.warning(f"Forecast is based on only {num_weeks_of_data} weeks of data. For a more reliable forecast, more historical data is recommended.")

        weights = np.arange(1, num_weeks_of_data + 1)
        normalized_weights = weights / np.sum(weights)
        
        return weekly_throughput, normalized_weights

    @staticmethod
    def create_when_forecast_chart(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str) -> Tuple[Optional[Figure], Optional[Dict[int, datetime]]]:
        """Creates the 'When' forecast chart by calling the cached simulation."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)

        if weekly_throughput is None or weekly_throughput.mean() == 0:
            return None, None

        with st.spinner(f"Running {Config.FORECASTING_SIMULATIONS} 'when' simulations for {items_to_complete} items..."):
            completion_weeks_data = ChartGenerator._run_when_simulation(items_to_complete, weekly_throughput, normalized_weights)

        if not completion_weeks_data: return None, None
        
        completion_days_data = [w * 7 for w in completion_weeks_data]
        value_counts = pd.Series(completion_days_data).value_counts().sort_index()

        fig = go.Figure(data=[go.Bar(name='Simulations', x=value_counts.index, y=value_counts.values)])
        fig.update_layout(
            title=f"Forecast: When Will We Finish {items_to_complete} Items?",
            xaxis_title=f"Days from {start_date.strftime('%d %b, %Y')} to Completion", 
            yaxis_title="Frequency (Number of Simulations)", 
            bargap=0.5
        )

        percentile_dates = {}
        for p in Config.PERCENTILES:
            days = np.percentile(completion_days_data, p)
            percentile_dates[p] = start_date + timedelta(days=int(days))
            fig.add_vline(
                x=days, line_dash="dash", line_color=Config.PERCENTILE_COLORS.get(p),
                annotation_text=f"{p}%", annotation_position="top right"
            )
        return fig, percentile_dates

    @staticmethod
    def run_when_scenario_forecast(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str) -> Optional[Dict]:
        """Runs the good week/bad week scenario analysis."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)

        if weekly_throughput is None or normalized_weights is None or weekly_throughput.mean() == 0:
            return None

        median_throughput = weekly_throughput.median()
        good_weeks = weekly_throughput[weekly_throughput > median_throughput]
        bad_weeks = weekly_throughput[weekly_throughput <= median_throughput]

        if good_weeks.empty or bad_weeks.empty: return None

        good_week_sim_results = ChartGenerator._run_when_simulation(items_to_complete, weekly_throughput, normalized_weights, good_weeks)
        bad_week_sim_results = ChartGenerator._run_when_simulation(items_to_complete, weekly_throughput, normalized_weights, bad_weeks)

        if not good_week_sim_results or not bad_week_sim_results: return None

        scenario_results = {'good_week': {}, 'bad_week': {}, 'median': median_throughput}
        for p in Config.PERCENTILES:
            good_days = np.percentile([w * 7 for w in good_week_sim_results], p)
            scenario_results['good_week'][p] = start_date + timedelta(days=int(good_days))
            
            bad_days = np.percentile([w * 7 for w in bad_week_sim_results], p)
            scenario_results['bad_week'][p] = start_date + timedelta(days=int(bad_days))
            
        return scenario_results
        
    @staticmethod
    def _prepare_chart_data(df: DataFrame, columns: List[str]) -> DataFrame:
        """Prepares a DataFrame for charting by formatting columns."""
        chart_df = df[columns].copy()
        for col in ['Completed date', 'Start date']:
            if col in chart_df.columns:
                chart_df[f'{col.replace(" ", "_")}_formatted'] = chart_df[col].dt.strftime('%d/%m/%Y')
        for col in ['Cycle time', 'Age']:
            if col in chart_df.columns:
                chart_df[f'{col.replace(" ", "_")}_formatted'] = chart_df[col].astype(int).astype(str)
        chart_df['Work type'] = pd.Categorical(chart_df['Work type'], categories=ChartGenerator._order_work_types(chart_df), ordered=True)
        return chart_df

    @staticmethod
    def _order_work_types(df: DataFrame) -> List[str]:
        """Returns a sorted list of unique work types based on the config order."""
        work_types = df['Work type'].unique()
        ordered = [wt for wt in Config.WORK_TYPE_ORDER if wt in work_types]
        ordered.extend(sorted(set(work_types) - set(Config.WORK_TYPE_ORDER)))
        return ordered

    @staticmethod
    def _add_trend_line(fig: Figure, data: DataFrame) -> None:
        """Adds a linear regression trend line to a chart."""
        if len(data) <= 2: return
        try:
            X = data['Date'].apply(lambda date: date.toordinal()).values.reshape(-1, 1)
            y = data["WIP"].values
            reg = LinearRegression().fit(X, y)
            trend_y = reg.predict(X)
            fig.add_trace(go.Scatter(x=data["Date"], y=trend_y, mode='lines', name='Trend', line=dict(color='red', dash='dash', width=2), hovertemplate=ChartConfig.TREND_LINE_HOVER))
        except Exception: pass


class StatsCalculator:
    """Calculates various statistics for the dashboard."""
    @staticmethod
    def summary_stats(df: DataFrame) -> Dict[str, int]:
        """Calculates total, completed, and in-progress item counts."""
        return {
            'total': len(df),
            'completed': df['Completed date'].notna().sum(),
            'in_progress': (df['Start date'].notna() & df['Completed date'].isna()).sum()
        }

    @staticmethod
    def cycle_time_stats(df: DataFrame) -> Optional[Dict[str, int]]:
        """Calculates average, median, and percentile cycle times."""
        completed = df.dropna(subset=['Cycle time'])
        if completed.empty: return None
        stats = {
            'average': int(completed['Cycle time'].mean()),
            'median': int(completed['Cycle time'].median())
        }
        for p in Config.PERCENTILES:
            stats[f'p{p}'] = int(np.percentile(completed['Cycle time'], p))
        return stats


class Dashboard:
    """The main class for the Streamlit dashboard application."""
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.raw_df, self.processed_df, self.filtered_df = None, None, None
        self.status_mapping = {}
        self.selections = {}

    def run(self):
        """Executes the main dashboard workflow."""
        with st.spinner("🔄 Processing JIRA export..."):
            loaded_df = DataProcessor.load_data(self.uploaded_file)
            if loaded_df is None: return
            self.raw_df = DataProcessor.clean_data(loaded_df)
        
        self.status_mapping = StatusManager.extract_status_columns(self.raw_df)
        if not self.status_mapping:
            st.error("No status columns found. Ensure JIRA export includes columns with '->' prefix.")
            return

        date_bounds_df = self._pre_process_for_sidebar()

        self._display_sidebar(date_bounds_df)
        
        start_col = self.selections.get("start_col")
        completed_col = self.selections.get("completed_col")
        
        if not start_col or not completed_col:
            st.info("ℹ️ Please select a 'Starting Status' and 'Done Status' from the Chart-Specific Controls to generate charts.")
            return
            
        is_valid, error_msg = StatusManager.validate_status_order(self.raw_df, start_col, completed_col)
        if not is_valid:
            st.error(error_msg)
            return

        with st.spinner("🔄 Processing with selected statuses..."):
            self.processed_df = DataProcessor.process_dates(self.raw_df, start_col, completed_col)
        
        if self.processed_df is None or self.processed_df.empty:
            st.error("No valid data found with the selected statuses.")
            return

        self.filtered_df = self._apply_all_filters(self.processed_df, apply_date_filter=True)
        self._display_header_and_metrics()
        self._display_charts()

    def _pre_process_for_sidebar(self) -> DataFrame:
        """A light processing step to get all possible dates for sidebar validation."""
        status_cols = list(StatusManager.extract_status_columns(self.raw_df).values())
        if not status_cols:
            return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})

        df = self.raw_df.copy()
        
        all_dates = []
        for col in status_cols:
            dates_in_col = df[col].apply(DataProcessor._extract_earliest_date).dropna()
            all_dates.extend(pd.to_datetime(dates_in_col, errors='coerce').dropna())

        if not all_dates:
             return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})

        date_df = pd.DataFrame(all_dates, columns=['Date'])
        return pd.DataFrame({
            'Start date': [date_df['Date'].min()],
            'Completed date': [date_df['Date'].max()]
        })

    def _display_sidebar(self, date_bounds_df: DataFrame):
        """Displays the sidebar for user configuration and filters."""
        st.sidebar.markdown("## ⚙️ Global Configuration")
        st.sidebar.caption("Settings that define the core dataset for all charts.")
        self._sidebar_global_filters(date_bounds_df)

        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 Chart-Specific Controls")
        st.sidebar.caption("Customize individual charts.")
        self._sidebar_chart_controls()

    def _sidebar_global_filters(self, date_bounds_df: DataFrame):
        """Controls for filtering the global dataset."""
        st.sidebar.markdown("#### 📋 Global Data Filters")
        self.selections["work_types"] = st.sidebar.multiselect("Work Item Type", options=["All"] + ChartGenerator._order_work_types(self.raw_df), default=["All"], help="Select one or more work types.") or ["All"]
        self.selections["date_range"] = st.sidebar.selectbox("Date Range", Config.DATE_RANGES, index=0)
        self.selections["custom_start_date"], self.selections["custom_end_date"] = None, None
        
        if self.selections["date_range"] == "Custom":
            min_date = date_bounds_df['Start date'].iloc[0]
            max_date = date_bounds_df['Completed date'].iloc[0]
            
            start_val = min_date if pd.notna(min_date) else datetime.now().date()
            end_val = max_date if pd.notna(max_date) else datetime.now().date()

            self.selections["custom_start_date"] = st.sidebar.date_input(
                "Start date",
                min_value=min_date if pd.notna(min_date) else None,
                max_value=max_date if pd.notna(max_date) else None,
                value=start_val
            )
            self.selections["custom_end_date"] = st.sidebar.date_input(
                "End date",
                min_value=min_date if pd.notna(min_date) else None,
                max_value=max_date if pd.notna(max_date) else None,
                value=end_val
            )

        st.sidebar.caption("Note: Date Range does not apply to the Work Item Age chart.")

        st.sidebar.markdown("#### Optional Filters")
        st.sidebar.caption("These will show depending on your data set")
        for f_name, f_type in Config.OPTIONAL_FILTERS.items():
            if f_name in self.raw_df.columns:
                unique_vals = self._get_unique_values(self.raw_df[f_name], f_type)
                if f_type == "single":
                    self.selections[f_name] = st.sidebar.selectbox(f_name, ["All"] + unique_vals, key=f"filter_{f_name}")
                else:
                    self.selections[f_name] = st.sidebar.multiselect(f_name, ["All"] + unique_vals, default=["All"], key=f"filter_{f_name}")

    def _sidebar_chart_controls(self):
        """Controls for customizing individual charts."""
        with st.sidebar.expander("📈 Cycle Time & WIP Charts", expanded=True):
            status_options = ["None"] + list(self.status_mapping.keys())
            self.selections["start_status"] = st.sidebar.selectbox(
                "Starting Status",
                status_options,
                help="Select when work starts. This defines the start of Cycle Time and when an item becomes Work in Progress (WIP)."
            )
            self.selections["completed_status"] = st.sidebar.selectbox(
                "Done Status",
                status_options,
                help="Select when work is completed. This defines the end of Cycle Time."
            )
            self.selections["start_col"] = self.status_mapping.get(self.selections["start_status"])
            self.selections["completed_col"] = self.status_mapping.get(self.selections["completed_status"])
        
        with st.sidebar.expander("📈 Cycle Time & Age Percentiles"):
            show_percentiles = st.checkbox("Show Percentile Lines", value=True)
            self.selections["percentiles"] = {f"show_{p}th": show_percentiles for p in Config.PERCENTILES}
            if show_percentiles:
                c1, c2 = st.columns(2)
                for p, col in zip(Config.PERCENTILES, [c1, c2, c1, c2]):
                    self.selections["percentiles"][f"show_{p}th"] = col.checkbox(f"{p}th", value=True, key=f"{p}th_visible")
        
    def _get_unique_values(self, series: pd.Series, filter_type: str) -> List[str]:
        """Gets unique values from a Series, handling multi-value strings."""
        if series.dropna().empty: return []
        if filter_type == "multi":
            return sorted(series.dropna().astype(str).str.split(',').explode().str.strip().unique())
        return sorted(series.dropna().unique())

    def _apply_all_filters(self, source_df: pd.DataFrame, apply_date_filter: bool) -> pd.DataFrame:
        """Applies all selected filters to the DataFrame."""
        df = source_df.copy()
        
        if "All" not in self.selections.get("work_types", ["All"]):
            df = df[df["Work type"].isin(self.selections["work_types"])]
        
        for f_name, f_type in Config.OPTIONAL_FILTERS.items():
            selection = self.selections.get(f_name)
            if selection and f_name in df.columns:
                if f_type == "single" and selection != "All":
                    df = df[df[f_name] == selection]
                elif f_type == "multi" and "All" not in selection:
                    pattern = '|'.join(re.escape(str(s)) for s in selection)
                    df = df[df[f_name].str.contains(pattern, na=False)]

        if apply_date_filter:
            df = self._apply_date_filter(df, self.selections["date_range"], self.selections["custom_start_date"], self.selections["custom_end_date"])
        
        return df

    def _apply_date_filter(self, df: pd.DataFrame, date_range: str, custom_start_date, custom_end_date) -> pd.DataFrame:
        """Filters the DataFrame based on the selected date range."""
        if date_range == "All time": return df
        today = pd.to_datetime(datetime.now().date())
        if date_range == "Last 30 days": cutoff = today - pd.DateOffset(days=30)
        elif date_range == "Last 60 days": cutoff = today - pd.DateOffset(days=60)
        elif date_range == "Last 90 days": cutoff = today - pd.DateOffset(days=90)
        elif date_range == "Custom" and custom_start_date and custom_end_date:
            start_date, end_date = pd.to_datetime(custom_start_date), pd.to_datetime(custom_end_date)
            return df[(df["Start date"] >= start_date) & ((df["Completed date"].isna()) | (df["Completed date"] <= end_date))]
        else: return df
        return df[(df["Start date"] >= cutoff) | ((df["Completed date"].isna()) | (df["Completed date"] >= cutoff))]

    def _calculate_forecast_days(self) -> int:
        """Calculates the number of days for the 'how many' forecast based on user selection."""
        range_selection = self.selections.get("forecast_range")
        if range_selection == "Next 30 days": return 30
        if range_selection == "Next 60 days": return 60
        if range_selection == "Next 90 days": return 90
        if range_selection == "Custom":
            custom_date = self.selections.get("forecast_custom_date")
            if custom_date:
                delta = (custom_date - datetime.now().date()).days
                return max(1, delta)
        return 30

    def _display_header_and_metrics(self):
        """Displays the main header, information boxes, and summary metrics."""
        st.success(f"✅ Processed {len(self.raw_df)} work items with {len(self.status_mapping)} status columns.")
        with st.expander("📋 Available Statuses", expanded=False):
            st.write(", ".join(self.status_mapping.keys()))
        st.info(f"📊 **Status Configuration:** Starting: **{self.selections['start_status']}** | Done: **{self.selections['completed_status']}**")
        stats = StatsCalculator.summary_stats(self.filtered_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Items in Filter", stats['total']); c2.metric("✅ Completed in Filter", stats['completed']); c3.metric("🔄 In Progress in Filter", stats['in_progress'])
        st.info("ℹ️ **Cycle Time Formula:** (Done Date - Starting Date) + 1 days")
        active_filters = [format_multiselect_display(self.selections['work_types'], 'Work types')]
        for f_name in Config.OPTIONAL_FILTERS.items():
            if f_name in self.selections:
                selection = self.selections[f_name]
                if (isinstance(selection, list) and "All" not in selection) or (isinstance(selection, str) and selection != "All"):
                    active_filters.append(format_multiselect_display(selection, f_name))
        date_range_display = f"from {self.selections['custom_start_date'].strftime('%Y-%m-%d')} to {self.selections['custom_end_date'].strftime('%Y-%m-%d')}" if self.selections['date_range'] == "Custom" and self.selections['custom_start_date'] and self.selections['custom_end_date'] else self.selections['date_range']
        active_filters.append(date_range_display)
        st.markdown(f"**🎯 Showing:** {' | '.join(active_filters)}")

    def _display_charts(self):
        """Displays the main chart area with tabs."""
        main_tabs = st.tabs(["📈 Cycle Time", "📊 Work Item Age", "🔄 WIP Trend", "📊 Throughput", "🔮 Throughput Forecast"])
        cycle_stats = StatsCalculator.cycle_time_stats(self.filtered_df)

        with main_tabs[0]: self._display_cycle_time_charts(cycle_stats)
        with main_tabs[1]: self._display_work_item_age_chart(cycle_stats)
        with main_tabs[2]: self._display_wip_chart()
        with main_tabs[3]: self._display_throughput_chart()
        with main_tabs[4]: self._display_forecast_charts()

    def _display_cycle_time_charts(self, cycle_stats):
        """Displays the Cycle Time charts and statistics."""
        st.header("Cycle Time Analysis"); st.markdown("Cycle time measures the total time from when work starts until it's completed.")
        if cycle_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average", f"{cycle_stats['average']} days"); c2.metric("Median (50th %ile)", f"{cycle_stats['median']} days"); c3.metric("85th Percentile", f"{cycle_stats['p85']} days"); c4.metric("95th Percentile", f"{cycle_stats['p95']} days")
        else:
            st.warning("⚠️ No items with both start and done dates to calculate cycle time statistics."); return
        ct_tabs = st.tabs(["Scatter Plot", "Distribution (Histogram)", "Time in Status"])
        with ct_tabs[0]:
            chart = ChartGenerator.create_cycle_time_chart(self.filtered_df, self.selections["percentiles"])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("⚠️ No items with both start and done dates for Cycle Time scatter plot.")
        with ct_tabs[1]:
            chart = ChartGenerator.create_cycle_time_histogram(self.filtered_df, self.selections["percentiles"])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("⚠️ No items to display in Cycle Time histogram.")
        with ct_tabs[2]:
            st.markdown("This chart shows the average time items spend in each status column of your raw data export.")
            status_cols = list(self.status_mapping.values())
            chart, chart_data = ChartGenerator.create_time_in_status_chart(self.raw_df, status_cols)
            if chart: 
                st.plotly_chart(chart, use_container_width=True)
                
                if chart_data is not None and not chart_data.empty:
                    st.divider()
                    num_statuses = len(chart_data)
                    cols = st.columns(min(num_statuses, 5), gap="small")
                    for i, row in chart_data.iterrows():
                        col = cols[i % 5]
                        col.metric(
                            label=row['Status'],
                            value=f"{int(row['Average Time (Days)'])} days"
                        )
            else: 
                st.warning("⚠️ Not enough data to calculate time in status.")

    def _display_work_item_age_chart(self, cycle_stats):
        """Displays the Work Item Age chart and its controls."""
        st.header("Work Item Age Analysis")
        st.markdown("This chart shows how old your current 'in progress' items are. Use the controls below to define the statuses on the x-axis.")

        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)

        with col1:
            self.selections["age_start_status"] = st.selectbox(
                "Start Status (for Age calculation)",
                status_options,
                help="Select the status to start aging from.",
                key="age_start"
            )
        with col2:
            self.selections["age_done_status"] = st.selectbox(
                "End Status",
                status_options,
                help="Select the last status to show on the chart's x-axis.",
                key="age_end"
            )

        self.selections["age_start_col"] = self.status_mapping.get(self.selections["age_start_status"])
        self.selections["age_done_col"] = self.status_mapping.get(self.selections["age_done_status"])
        
        st.divider()

        if self.selections["age_start_col"] and self.selections["age_done_col"]:
            if cycle_stats:
                age_df_source = self._apply_all_filters(self.processed_df, apply_date_filter=False)
                chart = ChartGenerator.create_work_item_age_chart(
                    age_df_source,
                    self.selections["age_start_col"],
                    self.selections["age_done_col"],
                    cycle_stats,
                    self.selections["percentiles"]
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("⚠️ No 'in progress' items found for Work Item Age chart with the selected statuses and filters.")
            else:
                st.warning("⚠️ Cycle time stats are not available to display on the Work Item Age chart.")
        else:
            st.info("ℹ️ Please select a Start and End Status above to generate the chart.")

    def _display_wip_chart(self):
        """Displays the WIP chart."""
        chart = ChartGenerator.create_wip_chart(self.filtered_df)
        if chart: st.plotly_chart(chart, use_container_width=True)
        else: st.warning("⚠️ No items with start dates for WIP chart.")

    def _display_throughput_chart(self):
        """Displays the Throughput chart and its controls."""
        st.header("Throughput")
        st.markdown("Throughput measures the number of work items completed per unit of time. Use the control below to change the time unit.")

        col1, col2 = st.columns(2)
        with col1:
            self.selections["throughput_interval"] = st.selectbox(
                "Interval",
                Config.THROUGHPUT_INTERVALS,
                key="throughput_interval_selector"
            )
        with col2:
            status_options = ["None"] + list(self.status_mapping.keys())
            
            if 'throughput_status_key' not in st.session_state:
                st.session_state.throughput_status_key = status_options[-1] if len(status_options) > 1 else "None"
            
            self.selections["throughput_status"] = st.selectbox(
                "Choose the throughput status",
                status_options,
                key="throughput_status_key"
            )

        self.selections['throughput_status_col'] = self.status_mapping.get(self.selections["throughput_status"])

        st.divider()

        chart = ChartGenerator.create_throughput_chart(
            self.filtered_df, 
            self.selections["throughput_interval"],
            self.selections['throughput_status_col']
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ No items with the selected throughput status for this chart.")

    def _display_forecast_charts(self):
        """Displays the Forecasting charts and controls."""
        st.header("Throughput Forecasting")
        
        throughput_status = self.selections.get('throughput_status', 'N/A')
        info_text = (
            "Uses historical throughput data to run Monte Carlo simulations and forecast future outcomes. "
            f"The forecast simulation is based on items reaching the **'{throughput_status}'** status selected on the Throughput chart."
        )
        st.info(info_text)
        
        forecast_tabs = st.tabs(["**How Many** (by date)", "**When** (by # of items)"])
        
        forecast_source_df = self._apply_all_filters(self.processed_df, apply_date_filter=False)
        throughput_status_col = self.selections.get('throughput_status_col')

        with forecast_tabs[0]:
            st.subheader("How many items can we complete by a certain date?")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                self.selections["forecast_range"] = st.selectbox("Forecast Timeframe", Config.FORECAST_DATE_RANGES, index=0, key="how_many_timeframe")
            
            self.selections["forecast_custom_date"] = None
            if self.selections["forecast_range"] == "Custom":
                with col2:
                    self.selections["forecast_custom_date"] = st.date_input("Forecast End Date", min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=30), key="how_many_custom_date")
            
            st.divider()
            
            forecast_days = self._calculate_forecast_days()
            chart = ChartGenerator.create_how_many_forecast_chart(forecast_source_df, forecast_days, throughput_status_col)
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("⚠️ Insufficient historical data for forecasting. Check that the selected Throughput Status has completed items.")

        with forecast_tabs[1]:
            st.subheader("When can we expect to complete a given number of items?")
            
            col1, col2 = st.columns(2)
            with col1:
                items_to_complete = st.number_input("Number of items to forecast:", min_value=1, value=20, step=1, key="when_forecast_items")
            with col2:
                forecast_start_date = st.date_input("Forecast start date", value=datetime.now().date(), key="when_forecast_start")

            st.divider()

            chart, stats = ChartGenerator.create_when_forecast_chart(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col)
            
            if stats:
                st.markdown("""
                <style>
                .forecast-metric-container { text-align: center; }
                .forecast-metric-label { font-size: 1.1em; font-weight: bold; color: #808495; }
                .forecast-metric-value { font-size: 2em; font-weight: bold; }
                </style>
                """, unsafe_allow_html=True)
                
                cols = st.columns(len(stats))
                for i, (p, date_val) in enumerate(stats.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="forecast-metric-container">
                            <div class="forecast-metric-label">{p}% Likelihood</div>
                            <div class="forecast-metric-value">{date_val.strftime("%d %b, %Y")}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if chart: 
                st.plotly_chart(chart, use_container_width=True)
            else: 
                st.warning("⚠️ Insufficient historical data for forecasting. Check that the selected Throughput Status has completed items.")

            with st.expander("🔍 Explore Forecast Scenarios"):
                st.markdown("This section explores how your completion date changes depending on whether your first week is 'good' (above median) or 'bad' (at or below median).")
                scenario_stats = ChartGenerator.run_when_scenario_forecast(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col)
                if scenario_stats:
                    st.subheader(f"Scenario 1: Good First Week (> {int(scenario_stats['median'])} items)")
                    cols = st.columns(len(scenario_stats['good_week']))
                    for i, (p, date_val) in enumerate(scenario_stats['good_week'].items()):
                        cols[i].metric(label=f"{p}% Likelihood", value=date_val.strftime("%d %b, %Y"))

                    st.subheader(f"Scenario 2: Bad First Week (≤ {int(scenario_stats['median'])} items)")
                    cols = st.columns(len(scenario_stats['bad_week']))
                    for i, (p, date_val) in enumerate(scenario_stats['bad_week'].items()):
                        cols[i].metric(label=f"{p}% Likelihood", value=date_val.strftime("%d %b, %Y"))
                else:
                    st.info("Not enough data to run scenario analysis.")

def display_welcome_message():
    """Displays the initial welcome message and instructions."""
    st.markdown("### 👋 Welcome to the Flow Metrics Dashboard!")
    st.markdown("A dynamic set of flow metrics charts and forecasting built to analyze data exported from the Jira plugins, Status Time Reports & Status Time Reports Free.")
    st.markdown("---")
    st.markdown("**Link to plugins:**")

    # Read and encode logos
    try:
        with open("Status time pro icon.png", "rb") as f:
            pro_logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <a href="https://marketplace.atlassian.com/apps/1221826/status-time-reports-time-in-status" target="_blank" style="text-decoration: none; color: inherit;">
                <img src="data:image/png;base64,{pro_logo_b64}" style="height: 1.2em; vertical-align: middle; margin-right: 8px;">
                Status Time Reports
            </a>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("- [Status Time Reports](https://marketplace.atlassian.com/apps/1221826/status-time-reports-time-in-status)")

    try:
        with open("Status time free icon.png", "rb") as f:
            free_logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <a href="https://marketplace.atlassian.com/apps/1222051/status-time-reports-free-time-in-status?hosting=cloud&tab=overview" target="_blank" style="text-decoration: none; color: inherit;">
                <img src="data:image/png;base64,{free_logo_b64}" style="height: 1.2em; vertical-align: middle; margin-right: 8px;">
                Status Time Reports Free
            </a>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("- [Status Time Reports Free](https://marketplace.atlassian.com/apps/1222051/status-time-reports-free-time-in-status?hosting=cloud&tab=overview)")

def format_multiselect_display(selection, name: str) -> str:
    """Formats a list from a multiselect for clean display."""
    if not selection or (isinstance(selection, list) and "All" in selection): return f"All {name}"
    if isinstance(selection, list):
        if len(selection) == 1: return selection[0]
        if len(selection) <= 3: return " & ".join(selection)
        return f"{', '.join(selection[:2])} & {len(selection)-2} more"
    return str(selection)

def main():
    """Main function to run the Streamlit app."""
    st.title("📊 Flow Metrics Dashboard")
    
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"], help="Upload a JIRA export CSV.")
    if uploaded_file:
        Dashboard(uploaded_file).run()
    else:
        display_welcome_message()

if __name__ == "__main__":
    main()