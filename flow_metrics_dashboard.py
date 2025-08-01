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
import unicodedata

def clean_value(value: any) -> str:
    """Removes common Jira suffixes like -DPID or -1234 from a string value."""
    if not isinstance(value, str):
        return str(value)
    # This regex removes a hyphen followed by uppercase letters and/or digits at the end of the string.
    return re.sub(r'-[A-Z0-9]+$', '', value.strip()).strip()

# Configuration
st.set_page_config(page_title="Flow Metrics Dashboard", layout="wide")

class ColorManager:
    """Manages color palettes for the dashboard."""
    DEFAULT_COLORS = {
        'Epic': '#8B5CF6', 'Story': '#10B981', 'Task': '#3B82F6',
        'Bug': '#EF4444', 'Spike': '#F97316'
    }
    COLOR_BLIND_FRIENDLY_COLORS = {
        'Epic': 'rgb(230,97,1)', 'Story': 'rgb(253,184,99)', 'Task': 'rgb(178,171,210)',
        'Bug': 'rgb(94,60,153)', 'Spike': '#b15928'
    }
    DEFAULT_PERCENTILE_COLORS = { 50: "red", 70: "orange", 85: "green", 95: "blue" }
    COLOR_BLIND_FRIENDLY_PERCENTILE_COLORS = {
        50: '#E69F00', 70: '#d95f02', 85: '#7570b3', 95: '#e7298a'
    }
    DEFAULT_FORECAST_BOX_COLORS = {
        50: "#f8d7da", 70: "#fff3cd", 85: "#d4edda", 95: "#a3bde0"
    }
    COLOR_BLIND_FRIENDLY_FORECAST_BOX_COLORS = {
        50: "#FADADD", 70: "#FFF8DC", 85: "#D4E6F1", 95: "#D1E8E2"
    }
    STABILITY_SCORE_COLORS = {
        "Stable": "#28a745", "Some Variability": "#ffc107", "High Variability": "#dc3545"
    }

    @staticmethod
    def get_work_type_colors(is_color_blind_mode: bool) -> Dict[str, str]:
        return ColorManager.COLOR_BLIND_FRIENDLY_COLORS if is_color_blind_mode else ColorManager.DEFAULT_COLORS

    @staticmethod
    def get_percentile_colors(is_color_blind_mode: bool) -> Dict[int, str]:
        return ColorManager.COLOR_BLIND_FRIENDLY_PERCENTILE_COLORS if is_color_blind_mode else ColorManager.DEFAULT_PERCENTILE_COLORS

    @staticmethod
    def get_forecast_box_colors(is_color_blind_mode: bool) -> Dict[int, str]:
        return ColorManager.COLOR_BLIND_FRIENDLY_FORECAST_BOX_COLORS if is_color_blind_mode else ColorManager.DEFAULT_FORECAST_BOX_COLORS

class Config:
    """Centralized configuration for the dashboard."""
    WORK_TYPE_ORDER = ['Epic', 'Story', 'Task', 'Bug', 'Spike']
    DATE_RANGES = ["All time", "Last 30 days", "Last 60 days", "Last 90 days", "Custom"]
    PERCENTILES = [50, 70, 85, 95]
    FORECAST_LIKELIHOODS = { 95: 5, 85: 15, 70: 30, 50: 50 }
    THROUGHPUT_INTERVALS = ["Weekly", "Fortnightly", "Monthly"]
    FORECASTING_SIMULATIONS = 10000
    FORECAST_DATE_RANGES = ["Next 30 days", "Next 60 days", "Next 90 days", "Custom"]
    FILTER_TYPE_HINTS = {
        "Team": "single", "Labels": "multi", "Components": "multi",
        "High Level Estimate-DPID": "multi", "RAG-DPID": "multi"
    }
    FILTER_EXCLUSIONS = ['Key', 'Summary', 'Created', 'Updated', 'Resolved', 'Last viewed']
    DEFAULT_COLOR = '#808080'

class ChartConfig:
    """Centralized configuration for chart templates and layouts."""
    CYCLE_TIME_HOVER = ("%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br><b>Completed:</b> %{customdata[2]}<br><b>Start:</b> %{customdata[3]}<br><b>Cycle time:</b> %{customdata[4]} days<extra></extra>")
    BUBBLE_CHART_HOVER = ("<b>Items:</b> %{marker.size}<br><b>Cycle Time:</b> %{y} days<br><b>Completed Date:</b> %{x|%d/%m/%Y}<br><br><b>Breakdown:</b><br>%{customdata[0]}<br><br><b>Keys:</b><br>%{customdata[1]}<extra></extra>")
    STORY_POINT_HOVER = ("<b>Key:</b> %{customdata[0]}<br><b>Cycle time:</b> %{y:.0f} days<br><b>Story Point:</b> %{x}<br><b>Date completed:</b> %{customdata[1]}<extra></extra>")
    AGE_CHART_HOVER = ("%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br><b>Status:</b> %{customdata[2]}<br><b>Start:</b> %{customdata[3]}<br><b>Age:</b> %{customdata[4]} days<extra></extra>")
    WIP_CHART_HOVER = "<b>Date:</b> %{x|%d/%m/%Y}<br><b>WIP count:</b> %{y}<br><b>Breakdown:</b><br>%{customdata[0]}<extra></extra>"
    THROUGHPUT_CHART_HOVER = ("<b>Period ending = %{customdata[0]}</b><br>Throughput = %{y} items<br>%{customdata[1]}<extra></extra>")
    TREND_LINE_HOVER = "<b>Trend Line</b><br>Date = %{x|%d/%m/%Y}<br>Trend value = %{y:.1f}<extra></extra>"

class StatusManager:
    """Handles status column extraction and validation."""
    @staticmethod
    @staticmethod
    def extract_status_columns(df: DataFrame) -> Dict[str, str]:
        # This simpler version uses the original, suffixed names as the keys, which is more robust.
        status_col_map = {
            col.replace("'->", "").strip(): col
            for col in df.columns if col.startswith("'->")
        }
        return status_col_map

    @staticmethod
    def validate_status_order(df: DataFrame, start_col: Optional[str], completed_col: Optional[str]) -> Tuple[bool, str]:
        if not start_col or not completed_col: return False, "Both start and completed statuses must be selected."
        if start_col == completed_col: return False, "Start and completed status cannot be the same."
        try:
            columns = list(df.columns)
            if columns.index(start_col) >= columns.index(completed_col): return False, "The selected 'Start Status' must come before the 'Done Status' in your workflow."
            return True, ""
        except ValueError:
            return False, "Invalid status columns selected."

class DataProcessor:
    """Handles loading and processing of JIRA export data."""
    @staticmethod
    @st.cache_data
    def load_data(uploaded_file) -> Optional[DataFrame]:
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, keep_default_na=False, encoding=encoding)
                df = df.dropna(how='all')
                if 'Issue type' in df.columns: df = df.rename(columns={'Issue type': 'Work type'})
                elif 'Issue Type' in df.columns: df = df.rename(columns={'Issue Type': 'Work type'})
                if not {'Key', 'Work type'}.issubset(df.columns):
                    st.error("Invalid File Format: Missing 'Key' or 'Work type' columns.")
                    return None
                return df
            except UnicodeDecodeError: continue
            except Exception as e:
                st.error(f"Error loading data with encoding {encoding}: {str(e)}")
                return None
        st.error("File Load Error: Could not read this file. Please ensure it's a standard CSV.")
        return None

    @staticmethod
    def clean_data(df: DataFrame) -> DataFrame:
        df_clean = df.copy()
        if df_clean.duplicated(subset=['Key']).any():
            duplicates = df_clean[df_clean.duplicated(subset=['Key'], keep=False)]
            st.warning(f"Data Quality: Found and removed {len(duplicates.drop_duplicates(subset=['Key']))} duplicate items. The first version of each was kept.")
        df_clean = df_clean.drop_duplicates(subset=['Key'], keep='first').copy()
        
        # Normalize characters first to handle any visual duplicates (e.g., different quote types)
        normalized_cols = [unicodedata.normalize('NFKC', col).strip() if isinstance(col, str) else col for col in df_clean.columns]
        # Then, clean the suffixes
        cleaned_cols = [clean_value(col) for col in normalized_cols]
        
        if len(cleaned_cols) != len(set(cleaned_cols)):
            st.error("File Error: Your CSV file has columns that become duplicates after removing suffixes (e.g., 'Labels-A' and 'Labels-B' both become 'Labels'). Please rename the columns in your source file to be unique before uploading.")
            return pd.DataFrame() 
            
        df_clean.columns = cleaned_cols

        for col in ['Key', 'Work type', 'Status']:
            if col in df_clean.columns: df_clean[col] = df_clean[col].apply(lambda x: unicodedata.normalize('NFKC', x).strip() if isinstance(x, str) else x)
        
        if 'Status' in df_clean.columns:
            df_clean['Status'] = df_clean['Status'].apply(clean_value)
            
        return df_clean

    @staticmethod
    @st.cache_data
    def process_dates(df: DataFrame, start_col: Optional[str], completed_col: Optional[str]) -> Optional[DataFrame]:
        try:
            processed_df = df.copy()
            
            start_dates = processed_df[start_col].apply(DataProcessor._extract_earliest_date) if start_col else pd.Series(pd.NaT, index=df.index)
            completed_dates = processed_df[completed_col].apply(DataProcessor._extract_latest_date) if completed_col else pd.Series(pd.NaT, index=df.index)
            
            processed_df['Start date'] = pd.to_datetime(start_dates, errors='coerce')
            processed_df['Completed date'] = pd.to_datetime(completed_dates, errors='coerce')

            return DataProcessor._calculate_cycle_time(processed_df)
        except Exception as e:
            st.error(f"Error processing dates: {str(e)}")
            return None

    @staticmethod
    def _parse_date_part(part: str) -> Optional[datetime]:
        part = part.strip()
        formats_to_try = ['%d-%b-%y %H:%M', '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y', '%Y-%m-%d']
        for fmt in formats_to_try:
            try: return datetime.strptime(part, fmt)
            except ValueError: continue
        dt = pd.to_datetime(part, errors='coerce')
        return dt if pd.notna(dt) else None

    @staticmethod
    def _extract_earliest_date(date_str: Union[str, float]) -> Optional[datetime]:
        if pd.isna(date_str) or str(date_str).strip() in ['-', '', 'nan']: return None
        valid_dates = [DataProcessor._parse_date_part(p) for p in str(date_str).split(',')]
        return min([d for d in valid_dates if d]) if any(valid_dates) else None

    @staticmethod
    def _extract_latest_date(date_str: Union[str, float]) -> Optional[datetime]:
        if pd.isna(date_str) or str(date_str).strip() in ['-', '', 'nan']: return None
        valid_dates = [DataProcessor._parse_date_part(p) for p in str(date_str).split(',')]
        return max([d for d in valid_dates if d]) if any(valid_dates) else None

    @staticmethod
    def _calculate_cycle_time(df: DataFrame) -> DataFrame:
        df = df[df['Work type'].notna() & (df['Work type'] != '')].copy()
        df['Cycle time'] = (df['Completed date'] - df['Start date']).dt.days + 1
        invalid_cycle = (df['Cycle time'].notna()) & (df['Cycle time'] < 1)
        if invalid_cycle.any():
            st.warning(f"Data Quality Warning: {invalid_cycle.sum()} item(s) excluded due to zero or negative cycle time.")
            df = df[~invalid_cycle]
        return df
        
    @staticmethod
    def _parse_elapsed_time_to_days(time_val: any) -> float:
        """Parses an elapsed time string (e.g., '3 days, 8 hours') or number into a total number of days."""
        if pd.isna(time_val):
            return np.nan
        
        if isinstance(time_val, (int, float)):
            return float(time_val)

        time_str = str(time_val).strip()
        if not time_str:
            return np.nan

        # If the value is a simple number, assume it's days
        if time_str.replace('.', '', 1).isdigit():
             return float(time_str)
        
        days = 0.0
        # Regex to find numbers and their units (d for day, h for hour, m for minute)
        parts = re.findall(r'(\d+\.?\d*)\s*(d|h|m)', time_str, re.IGNORECASE)
        
        for value, unit in parts:
            value = float(value)
            if unit.lower() == 'd':
                days += value
            elif unit.lower() == 'h':
                days += value / 24.0
            elif unit.lower() == 'm':
                days += value / (24.0 * 60.0)
        return days if days > 0 else np.nan

    @staticmethod
    @st.cache_data
    def calculate_flow_efficiency(df: DataFrame, all_statuses_in_cycle: List[str], wait_statuses: List[str]) -> DataFrame:
        df_eff = df.copy()

        # This function now expects the raw, suffixed column names
        # We will clean them here for the calculation
        
        # Create a map of {raw_col_name: clean_col_name}
        col_to_clean_map = {col: clean_value(col) for col in all_statuses_in_cycle}

        # Parse all relevant elapsed time columns into days
        for col_name in all_statuses_in_cycle:
            if col_name in df_eff.columns:
                clean_name = col_to_clean_map[col_name]
                df_eff[f"{clean_name}_days"] = df_eff[col_name].apply(DataProcessor._parse_elapsed_time_to_days)

        # Define which columns are active vs. wait based on user selection of CLEAN names
        active_statuses = [clean_value(s) for s in all_statuses_in_cycle if clean_value(s) not in wait_statuses]
        active_cols_days = [f"{s}_days" for s in active_statuses if f"{s}_days" in df_eff.columns]
        wait_cols_days = [f"{s}_days" for s in wait_statuses if f"{s}_days" in df_eff.columns]
        
        # Sum the columns to get Active Time and Wait Time
        df_eff['Active Time Days'] = df_eff[active_cols_days].sum(axis=1)
        df_eff['Wait Time Days'] = df_eff[wait_cols_days].sum(axis=1)
        # Calculate Cycle Time as the sum of all time spent in the defined workflow
        df_eff['Cycle time'] = df_eff['Active Time Days'] + df_eff['Wait Time Days']
        
        # Calculate Flow Efficiency, handling division by zero
        df_eff['Flow Efficiency'] = 0.0
        mask = df_eff['Cycle time'] > 0
        df_eff.loc[mask, 'Flow Efficiency'] = (df_eff.loc[mask, 'Active Time Days'] / df_eff.loc[mask, 'Cycle time']) * 100
        df_eff['Flow Efficiency'] = df_eff['Flow Efficiency'].clip(0, 100)
        
        return df_eff

class ChartGenerator:
    """Generates Plotly charts for the dashboard."""
    @staticmethod
    @st.cache_data
    def create_cumulative_flow_diagram(df: DataFrame, selected_statuses: List[str], status_col_map: Dict[str, str], date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime], is_color_blind_mode: bool) -> Optional[Figure]:
        if not selected_statuses or df.empty: return None
        status_cols = [status_col_map[s] for s in selected_statuses]
        cfd_df_melted = df.melt(id_vars=['Key'], value_vars=status_cols, var_name='Status Column', value_name='Date Value')
        cfd_df_melted['Date'] = cfd_df_melted['Date Value'].apply(DataProcessor._extract_earliest_date)
        cfd_df_melted.dropna(subset=['Date'], inplace=True)
        cfd_df_melted['Date'] = cfd_df_melted['Date'].dt.normalize()
        status_name_map = {v: k for k, v in status_col_map.items()}
        cfd_df_melted['Status'] = cfd_df_melted['Status Column'].map(status_name_map)
        if cfd_df_melted.empty: return None
        cfd_df_filtered = _apply_date_filter(cfd_df_melted, 'Date', date_range, custom_start_date, custom_end_date)
        if cfd_df_filtered.empty: return None
        daily_counts = cfd_df_filtered.groupby([cfd_df_filtered['Date'].dt.date, 'Status']).size().reset_index(name='Count')
        if daily_counts.empty: return None
        min_chart_date, max_chart_date = daily_counts['Date'].min(), daily_counts['Date'].max()
        full_date_range = pd.to_datetime(pd.date_range(start=min_chart_date, end=max_chart_date))
        cfd_pivot = daily_counts.pivot_table(index='Date', columns='Status', values='Count', fill_value=0)
        cfd_cumulative = cfd_pivot.cumsum().reindex(full_date_range.date, method='ffill').fillna(0)
        plot_df = cfd_cumulative.reset_index().rename(columns={'index': 'Date'}).melt(id_vars='Date', value_name='Count', var_name='Status')
        plot_df['Status'] = pd.Categorical(plot_df['Status'], categories=selected_statuses, ordered=True)
        color_map = ColorManager.get_work_type_colors(is_color_blind_mode) if is_color_blind_mode else None
        fig = px.area(plot_df, x='Date', y='Count', color='Status', title='Cumulative Flow Diagram', labels={'Date': 'Date', 'Count': 'Cumulative Count of Items'}, category_orders={'Status': selected_statuses}, color_discrete_map=color_map)
        
        # FIX: Reverse the order of the legend items to match the chart
        fig.update_layout(height=600, legend_title='Workflow Stage', legend_traceorder='reversed')
        
        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_chart(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        completed_df = df.dropna(subset=['Start date', 'Completed date', 'Cycle time'])
        if completed_df.empty: return None
        chart_df = ChartGenerator._prepare_chart_data(completed_df, ['Key', 'Work type', 'Completed date', 'Start date', 'Cycle time'])
        chart_df['Cycle_time_jittered'] = chart_df['Cycle time'] + np.random.uniform(-0.4, 0.4, size=len(chart_df))
        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scattergl(x=df_type['Completed date'], y=df_type['Cycle_time_jittered'], mode='markers', name=work_type, marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7), customdata=df_type[['Key', 'Work type', 'Completed_date_formatted', 'Start_date_formatted', 'Cycle_time_formatted']], hovertemplate=ChartConfig.CYCLE_TIME_HOVER))
        fig.update_layout(title="Cycle Time Scatterplot", xaxis_title="Completed Date", yaxis_title="Cycle Time (Days)", height=675, legend_title="Work Type")
        ChartGenerator._add_percentile_lines(fig, chart_df, 'Cycle time', chart_df["Completed date"], percentile_settings, is_color_blind_mode)
        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_bubble_chart(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        completed_df = df.dropna(subset=['Completed date', 'Cycle time', 'Work type'])
        if completed_df.empty: return None
        agg_df = completed_df.groupby(['Completed date', 'Cycle time']).agg(item_count=('Key', 'size'), keys=('Key', lambda x: '<br>'.join(x)), breakdown=('Work type', lambda s: '<br>'.join(f"{n}: {c}" for n, c in s.value_counts().items()))).reset_index()
        fig = go.Figure(data=[go.Scatter(x=agg_df['Completed date'], y=agg_df['Cycle time'], mode='markers+text', text=agg_df['item_count'], textposition='middle center', textfont=dict(color='white', size=10), marker=dict(size=agg_df['item_count'], sizemode='area', sizeref=2.*agg_df['item_count'].max()/(40.**2), sizemin=4, color='#3B82F6', opacity=0.7), customdata=agg_df[['breakdown', 'keys']], hovertemplate=ChartConfig.BUBBLE_CHART_HOVER)])
        fig.update_layout(title="Aggregated Cycle Time Bubble Chart", xaxis_title="Completed Date", yaxis_title="Cycle Time (Days)", height=900)
        ChartGenerator._add_percentile_lines(fig, completed_df, 'Cycle time', agg_df["Completed date"], percentile_settings, is_color_blind_mode)
        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_box_plot(df: DataFrame, interval: str, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        df_completed = df.dropna(subset=['Completed date', 'Cycle time']).copy()
        if df_completed.empty: return None
        freq_map = {'Weekly': 'W-MON', 'Monthly': 'M'}
        df_completed['Period'] = df_completed['Completed date'].dt.to_period(freq_map[interval]).dt.start_time
        fig = px.box(df_completed, x='Period', y='Cycle time', title=f'Cycle Time Distribution per {interval}', labels={'Period': interval, 'Cycle time': 'Cycle Time (Days)'}, points='all')
        fig.update_xaxes(tickformat="%b %Y" if interval == 'Monthly' else "%d %b %Y")
        fig.update_layout(height=900)
        ChartGenerator._add_percentile_lines(fig, df_completed, 'Cycle time', df_completed["Period"], percentile_settings, is_color_blind_mode, add_annotation=True)
        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_histogram(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        completed_df = df.dropna(subset=['Cycle time'])
        if completed_df.empty: return None
        fig = px.histogram(completed_df, x="Cycle time", title="Cycle Time Distribution", labels={'Cycle time': 'Cycle Time (Days)', 'count': 'Number of Work Items'}, color_discrete_sequence=['#3B82F6'])
        fig.update_layout(bargap=0.1, yaxis_title="Number of Work Items", height=600)
        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for p, color in percentile_colors.items():
            if percentile_settings.get(f"show_{p}th", True):
                val = np.percentile(completed_df['Cycle time'], p)
                fig.add_vline(x=val, line_dash="dash", line_color=color, annotation_text=f"{p}th: {int(val)}d", annotation_position="top right")
        return fig

    @staticmethod
    @st.cache_data
    def create_time_in_status_chart(df: DataFrame, status_cols: List[str]) -> Tuple[Optional[Figure], Optional[DataFrame]]:
        # Get the clean status names from the -> prefixed columns
        clean_status_names = [s.replace("'->", "").strip() for s in status_cols]
        
        all_durations = []
        # Loop through all statuses EXCEPT the last one
        for status_name in clean_status_names[:-1]:
            # Check if a corresponding non-prefixed elapsed time column exists
            if status_name in df.columns:
                durations = df[status_name].apply(DataProcessor._parse_elapsed_time_to_days).dropna()
                if not durations.empty:
                    avg_duration = durations.mean()
                    all_durations.append({'Status': status_name, 'Average Time (Days)': avg_duration})

        if not all_durations: return None, None
        
        chart_df = pd.DataFrame(all_durations)
        # Round the values for display
        chart_df['Average Time (Days)'] = np.ceil(chart_df['Average Time (Days)'])
        
        fig = px.bar(chart_df, x='Status', y='Average Time (Days)', title='Average Time in Each Status', text='Average Time (Days)')
        fig.update_traces(texttemplate='%{text:.0f}d', textposition='outside', textfont_size=14)
        fig.update_layout(yaxis_title="Average Time (Days)", font=dict(size=14), height=600, yaxis_range=[0, chart_df['Average Time (Days)'].max() * 1.15])
        return fig, chart_df
        
    @staticmethod
    @st.cache_data
    def create_85th_time_in_status_chart(df: DataFrame, status_cols: List[str]) -> Tuple[Optional[Figure], Optional[DataFrame]]:
        # Get the clean status names from the -> prefixed columns
        clean_status_names = [s.replace("'->", "").strip() for s in status_cols]
        
        all_durations = []
        # Loop through all statuses EXCEPT the last one
        for status_name in clean_status_names[:-1]:
            # Check if a corresponding non-prefixed elapsed time column exists
            if status_name in df.columns:
                durations = df[status_name].apply(DataProcessor._parse_elapsed_time_to_days).dropna()
                if not durations.empty:
                    p85_duration = durations.quantile(0.85)
                    all_durations.append({'Status': status_name, '85th Percentile Time (Days)': p85_duration})
        
        if not all_durations: return None, None

        chart_df = pd.DataFrame(all_durations)
        # Round the values for display
        chart_df['85th Percentile Time (Days)'] = np.ceil(chart_df['85th Percentile Time (Days)'])
        
        fig = px.bar(chart_df, x='Status', y='85th Percentile Time (Days)', title='85th Percentile Time in Each Status', text='85th Percentile Time (Days)')
        fig.update_traces(texttemplate='%{text:.0f}d', textposition='outside', textfont_size=14)
        fig.update_layout(yaxis_title="85th Percentile Time (Days)", font=dict(size=14), height=600, yaxis_range=[0, chart_df['85th Percentile Time (Days)'].max() * 1.15])
        return fig, chart_df

    @staticmethod
    @st.cache_data
    def create_flow_efficiency_histogram(df: DataFrame) -> Optional[Figure]:
        df_eff = df.dropna(subset=['Flow Efficiency'])
        if df_eff.empty: return None
        fig = px.histogram(df_eff, x="Flow Efficiency", title="Flow Efficiency Distribution", labels={'Flow Efficiency': 'Flow Efficiency (%)', 'count': 'Number of Work Items'}, color_discrete_sequence=['#3B82F6'])
        fig.update_layout(bargap=0.1, xaxis=dict(ticksuffix="%"), yaxis_title="Number of Work Items", height=600)
        return fig

    @staticmethod
    def create_work_item_age_chart(plot_df: DataFrame, wip_counts_by_status: Dict[str, int], status_order: List[str], cycle_time_percentiles: Dict[str, int], percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        age_data = []
        # The plot_df already contains items with a valid start date.
        for _, row in plot_df.iterrows():
            start_date_val = row.get('Start date')
            if pd.notna(start_date_val):
                age_data.append({
                    'Key': row['Key'],
                    'Work type': row['Work type'],
                    'Status': row['Status'],
                    'Age': (datetime.now() - start_date_val).days + 1,
                    'Start date': start_date_val
                })

        chart_df = pd.DataFrame(age_data) if age_data else pd.DataFrame(columns=['Key', 'Work type', 'Status', 'Age', 'Start date'])
        if not chart_df.empty:
            status_map = {status: i for i, status in enumerate(status_order)}
            chart_df['Status_Num'] = chart_df['Status'].map(status_map)
            if not chart_df.empty:
                 chart_df['Status_Jittered'] = chart_df['Status_Num'] + np.random.uniform(-0.25, 0.25, size=len(chart_df))
            chart_df = ChartGenerator._prepare_chart_data(chart_df, ['Key', 'Work type', 'Status', 'Age', 'Status_Jittered', 'Start date'])
            chart_df.dropna(subset=['Status', 'Status_Jittered'], inplace=True)

        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()

        if not chart_df.empty:
            for work_type in ChartGenerator._order_work_types(chart_df):
                df_type = chart_df[chart_df['Work type'] == work_type]
                fig.add_trace(go.Scattergl(x=df_type['Status_Jittered'], y=df_type['Age'], mode='markers', name=work_type, marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7), customdata=df_type[['Key', 'Work type', 'Status', 'Start_date_formatted', 'Age_formatted']], hovertemplate=ChartConfig.AGE_CHART_HOVER))
        
        max_age_plot = chart_df['Age'].max() if not chart_df.empty else 10
        max_percentile = max(cycle_time_percentiles.values()) if cycle_time_percentiles else 0
        y_axis_max = max(max_age_plot, max_percentile) * 1.15 if max(max_age_plot, max_percentile) > 0 else 10

        fig.update_layout(title="Work Item Age Analysis", yaxis_title="<b>Age (Calendar Days)</b>", height=675, legend_title="Work Type", xaxis=dict(title_text="", tickvals=list(range(len(status_order))), ticktext=[f"<b>{s}</b>" for s in status_order], tickfont=dict(size=14), showgrid=False, zeroline=False), yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey', tickfont=dict(size=14), title_font=dict(size=16), range=[0, y_axis_max]), showlegend=True)
        
        for i, status in enumerate(status_order):
            if i > 0: fig.add_vline(x=i - 0.5, line_width=2, line_color='LightGrey')
            count = wip_counts_by_status.get(status, 0)
            fig.add_annotation(x=i, y=y_axis_max, text=f"<b>WIP = {count}</b>", showarrow=False, font=dict(size=14, color="black"), yanchor="bottom", yshift=5)
        
        if cycle_time_percentiles:
            percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
            x_range_for_lines = [-0.5, len(status_order) - 0.5]
            for p_val in Config.PERCENTILES:
                key = f'p{p_val}'
                if percentile_settings.get(f"show_{p_val}th", True) and key in cycle_time_percentiles:
                    y_value = cycle_time_percentiles[key]
                    hover_text = f"<b>{p_val}th Percentile (from Cycle Time)</b><br>Value: {int(y_value)} days<br><i>{p_val}% of items finish in this time or less.</i>"
                    ChartGenerator._add_hoverable_line(fig, y_value, x_range_for_lines, hover_text, percentile_colors.get(p_val), f"{p_val}th: {int(y_value)}d")
        return fig

    @staticmethod
    @st.cache_data
    def create_story_point_chart(df: DataFrame, is_color_blind_mode: bool) -> Optional[Figure]:
        sp_col_name = None
        potential_sp_cols = ['story points', 'story point estimate']
        for col in df.columns:
            if col.strip().lower() in potential_sp_cols:
                if pd.to_numeric(df[col], errors='coerce').notna().any():
                    sp_col_name = col
                    break
        
        if not sp_col_name: return None
        
        df_sp = df.dropna(subset=['Cycle time', sp_col_name]).copy()
        df_sp = df_sp[pd.to_numeric(df_sp[sp_col_name], errors='coerce').notna()]
        df_sp[sp_col_name] = pd.to_numeric(df_sp[sp_col_name])
        if df_sp.empty: return None
        chart_df = ChartGenerator._prepare_chart_data(df_sp, ['Key', 'Work type', 'Completed date', 'Start date', 'Cycle time'])
        chart_df[sp_col_name] = df_sp[sp_col_name]
        chart_df['Cycle_time_jittered'] = chart_df['Cycle time'] + np.random.uniform(-0.4, 0.4, size=len(chart_df))
        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scattergl(x=df_type[sp_col_name], y=df_type['Cycle_time_jittered'], mode='markers', name=work_type, marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7), customdata=df_type[['Key', 'Completed_date_formatted']], hovertemplate=ChartConfig.STORY_POINT_HOVER))
        all_ticks = [1, 2, 3, 5, 8, 13, 20, 40, 100]
        max_sp_value = chart_df[sp_col_name].max()
        visible_ticks = [t for t in all_ticks if t <= max_sp_value]
        if not visible_ticks or max_sp_value > visible_ticks[-1]:
             if not any(abs(max_sp_value - t) < 0.1 for t in visible_ticks):
                visible_ticks.append(int(np.ceil(max_sp_value)))
                visible_ticks.sort()
        fig.update_layout(title="Story Point Correlation", xaxis_title="Story Points", yaxis_title="Cycle Time (Days)", height=675, legend_title="Work Type", legend=dict(yanchor="top", y=1, xanchor="left", x=1.02))
        fig.update_xaxes(tickmode='array', tickvals=visible_ticks, ticktext=[f"<b>{t}</b>" for t in visible_ticks], tickfont=dict(size=14))
        return fig

    @staticmethod
    def _add_percentile_lines(fig: Figure, df: pd.DataFrame, y_col: str, x_data, percentile_settings: Dict[str, bool], is_color_blind_mode: bool, add_annotation: bool = False):
        if df.empty: return
        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for p, color in percentile_colors.items():
            if percentile_settings.get(f"show_{p}th", True):
                y_value = np.percentile(df[y_col], p)
                hover_text = f"<b>{p}th Percentile</b><br>Value: {int(y_value)} days<br><i>{p}% of items finish in this time or less.</i>"
                if add_annotation:
                    fig.add_hline(y=y_value, line_dash="dash", line_color=color, annotation_text=f"{p}th: {int(y_value)}d", annotation_position="top left")
                else:
                    ChartGenerator._add_hoverable_line(fig, y_value, x_data, hover_text, color, f"{p}th: {int(y_value)}d")

    @staticmethod
    def _add_hoverable_line(fig, y_value, x_data, hover_text, color, annotation_text):
        fig.add_hline(y=y_value, line_dash="dash", line_color=color, line_width=1.5, annotation_text=annotation_text, annotation_position="top left")
        fig.add_trace(go.Scatter(x=list(x_data), y=[y_value] * len(x_data), mode='lines', line=dict(color='rgba(0,0,0,0)', width=20), hovertemplate=hover_text + "<extra></extra>", showlegend=False))

    @staticmethod
    @st.cache_data
    def create_wip_chart(df: DataFrame, date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime]) -> Optional[Figure]:
        if df is None or df.empty: return None
        wip_df = df.dropna(subset=['Start date'])
        if wip_df.empty: return None
        plot_min_date = wip_df['Start date'].min()
        latest_start = wip_df['Start date'].max()
        latest_completion = wip_df['Completed date'].max()
        plot_max_date = max(dt for dt in [latest_start, latest_completion] if pd.notna(dt))
        today = pd.to_datetime(datetime.now().date())
        if plot_max_date > today: plot_max_date = today
        all_dates = pd.date_range(start=plot_min_date, end=plot_max_date, freq='D')
        filtered_dates_df = _apply_date_filter(pd.DataFrame({'Date': all_dates}), 'Date', date_range, custom_start_date, custom_end_date)
        if filtered_dates_df.empty: return None
        filtered_dates = filtered_dates_df['Date']
        daily_wip_data = []
        for day in filtered_dates:
            daily_wip_df = wip_df[(wip_df['Start date'] <= day) & ((wip_df['Completed date'].isna()) | (wip_df['Completed date'] > day))]
            breakdown_str = '<br>'.join(f"{wt}: {count}" for wt, count in daily_wip_df['Work type'].value_counts().items())
            daily_wip_data.append({'Date': day, 'WIP': len(daily_wip_df), 'Breakdown': breakdown_str})
        if not daily_wip_data: return None
        wip_over_time = pd.DataFrame(daily_wip_data)
        fig = px.line(wip_over_time, x="Date", y="WIP", title="WIP (Work In Progress) Run Chart")
        fig.update_traces(customdata=wip_over_time[['Breakdown']], hovertemplate=ChartConfig.WIP_CHART_HOVER)
        fig.update_layout(height=600)
        ChartGenerator._add_trend_line(fig, wip_over_time)
        return fig

    @staticmethod
    @st.cache_data
    def create_throughput_chart(df: DataFrame, interval: str, throughput_status_col: str, date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime], sprint_anchor_date: Optional[datetime.date] = None, overall_max_date: Optional[datetime] = None) -> Optional[Tuple[Figure, DataFrame]]:
        if not throughput_status_col: return None, None
        throughput_df = df.copy()
        throughput_df['Throughput Date'] = throughput_df[throughput_status_col].apply(DataProcessor._extract_latest_date)
        throughput_df.dropna(subset=['Throughput Date'], inplace=True)
        if throughput_df.empty: return None, None

        # Logic for Fortnightly sprints remains custom
        if interval == 'Fortnightly':
            if not sprint_anchor_date:
                st.warning("For fortnightly throughput, please select a 'Sprint End Date' to set the 2-week cycle.")
                return None, None
            anchor = pd.to_datetime(sprint_anchor_date)
            min_date, max_date_in_data = throughput_df['Throughput Date'].min(), throughput_df['Throughput Date'].max()
            bins = [anchor]
            temp_date_back = anchor
            while temp_date_back > min_date:
                temp_date_back -= timedelta(days=14)
                bins.insert(0, temp_date_back)
            
            temp_date_forward = anchor
            while temp_date_forward < max_date_in_data:
                temp_date_forward += timedelta(days=14)
                bins.append(temp_date_forward)
            
            bins = sorted(list(set(bins)))
            
            # Adjust bin edges to be the end of the day (23:59:59)
            adjusted_bins = [pd.to_datetime(edge).normalize() + pd.Timedelta(days=1, nanoseconds=-1) for edge in bins]
            
            throughput_df['Period End'] = pd.cut(throughput_df['Throughput Date'], bins=adjusted_bins, right=True, include_lowest=True, labels=bins[1:])
            throughput_df.dropna(subset=['Period End'], inplace=True)
            
            agg_df = throughput_df.groupby('Period End').agg(
                Throughput=('Key', 'count'), 
                Details=('Work type', lambda s: '<br>'.join(f"{wt}: {count}" for wt, count in s.value_counts().items()))
            ).reset_index()

            agg_df['Period End'] = pd.to_datetime(agg_df['Period End'])
            agg_df['Period Start'] = agg_df['Period End'] - pd.DateOffset(days=13)
        
        else:
            # Use the robust pd.Grouper for standard Weekly and Monthly intervals
            if interval == 'Weekly':
                freq_string = 'W-SUN'  # Use Sunday-ending weeks
            else:  # Monthly
                freq_string = 'M'  # Use Month-end frequency
            
            agg_df = throughput_df.groupby(pd.Grouper(key='Throughput Date', freq=freq_string)).agg(
                Throughput=('Key', 'count'), 
                Details=('Work type', lambda s: '<br>'.join(f"{wt}: {count}" for wt, count in s.value_counts().items()))
            ).reset_index()
            
            agg_df.rename(columns={'Throughput Date': 'Period End'}, inplace=True)

            if interval == 'Weekly':
                agg_df['Period Start'] = agg_df['Period End'] - pd.DateOffset(days=6)
            else:  # Monthly
                agg_df['Period Start'] = agg_df['Period End'].apply(lambda d: d.replace(day=1))
        
        # --- Common logic for all intervals ---
        agg_df = _apply_date_filter(agg_df, 'Period End', date_range, custom_start_date, custom_end_date)
        if agg_df.empty or (overall_max_date and agg_df[agg_df['Period Start'] <= overall_max_date].empty): return None, None
        if overall_max_date: agg_df = agg_df[agg_df['Period Start'] <= overall_max_date]
        
        agg_df = agg_df.sort_values(by='Period Start')
        agg_df['Period_End_Formatted'] = agg_df['Period End'].dt.strftime('%d/%m/%Y')
        agg_df['Details'] = "<b>Breakdown:</b><br>" + agg_df['Details']
        
        if interval == 'Monthly':
            agg_df['Period Label'] = agg_df['Period End'].dt.strftime('%b %Y')
        else:
            agg_df['Period Label'] = agg_df['Period End'].dt.strftime('%d %b %Y')

        fig = px.bar(agg_df, x='Period Label', y="Throughput", title=f"Throughput per {interval.replace('ly', '')}", text="Throughput")
        fig.update_traces(textposition='outside', hovertemplate=ChartConfig.THROUGHPUT_CHART_HOVER, customdata=agg_df[['Period_End_Formatted', 'Details']].values)
        fig.update_layout(height=600, yaxis_range=[0, agg_df['Throughput'].max() * 1.15], xaxis_title="Period Ending", xaxis_categoryorder="array", xaxis_categoryarray=agg_df['Period Label'].tolist())
        
        return fig, agg_df

    @staticmethod
    @st.cache_data(show_spinner="Running simulations...")
    def _get_recent_weekly_throughput(df: DataFrame, status_col: str) -> Tuple[Optional[pd.Series], Optional[np.ndarray]]:
        if not status_col: return None, None
        forecast_df = df.copy()
        forecast_df['Forecast Completion Date'] = forecast_df[status_col].apply(DataProcessor._extract_latest_date)
        completed_df = forecast_df.dropna(subset=['Forecast Completion Date'])
        if len(completed_df) < 2: return None, None
        last_completion_date = completed_df['Forecast Completion Date'].max()
        recent_completed_df = completed_df[completed_df['Forecast Completion Date'] > last_completion_date - pd.DateOffset(weeks=25)]
        if recent_completed_df.empty:
            st.warning("Forecasting requires more data: at least two weeks of completed work from the last 25 weeks.")
            return None, None
        weekly_throughput = recent_completed_df.groupby(pd.Grouper(key='Forecast Completion Date', freq='W-MON')).size()
        weekly_throughput = weekly_throughput[weekly_throughput.index <= recent_completed_df['Forecast Completion Date'].max()]
        num_weeks = len(weekly_throughput)
        if num_weeks < 2:
            st.warning("Forecasting requires more data: at least two weeks of completed work from the last 25 weeks.")
            return None, None
        if num_weeks < 7: st.warning(f"Note: Forecast is based on only {num_weeks} weeks of data.")
        weights = np.arange(1, num_weeks + 1)
        return weekly_throughput, weights / np.sum(weights)

    @staticmethod
    @st.cache_data(show_spinner="Running 'How Many' simulations...")
    def create_how_many_forecast_chart(df: DataFrame, forecast_days: int, throughput_status_col: str, is_color_blind_mode: bool) -> Optional[Figure]:
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None: return None
        num_weeks, fractional_week = divmod(forecast_days, 7)
        simulations = np.random.choice(weekly_throughput, size=(Config.FORECASTING_SIMULATIONS, num_weeks), replace=True, p=normalized_weights)
        forecast_counts = simulations.sum(axis=1)
        if fractional_week > 0:
            last_week_sim = np.random.choice(weekly_throughput, size=Config.FORECASTING_SIMULATIONS, replace=True, p=normalized_weights)
            forecast_counts += (last_week_sim * (fractional_week / 7.0)).astype(int)
        counts, bin_edges = np.histogram(forecast_counts, bins=30, range=(forecast_counts.min(), forecast_counts.max()))
        fig = go.Figure(data=[go.Bar(x=(bin_edges[:-1] + bin_edges[1:]) / 2, y=counts, name='Simulations')])
        fig.update_layout(title=f"Forecast: How Many Items in the Next {forecast_days} Days?", xaxis_title="Number of Items Completed", yaxis_title="Frequency", bargap=0.1, yaxis_range=[0, counts.max() * 1.20], height=600)
        summary_text = f"**Forecast Summary (for next {forecast_days} days):**"
        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for likelihood, percentile in sorted(Config.FORECAST_LIKELIHOODS.items(), reverse=True):
            value = np.percentile(forecast_counts, percentile)
            color_key = next((k for k, v in Config.FORECAST_LIKELIHOODS.items() if v == percentile), 50)
            fig.add_vline(x=value, line_dash="dash", line_color=percentile_colors.get(color_key), annotation_text=f"{likelihood}%: {int(value)}", annotation_position="top left")
            summary_text += f"\n- There is a **{likelihood}% chance** to complete **{int(value)} or more** items."
        st.markdown(summary_text)
        return fig

    @staticmethod
    @st.cache_data(show_spinner="Running 'When' simulations...")
    def create_when_forecast_chart(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str, scope_complexity: str, team_focus: str, is_color_blind_mode: bool) -> Tuple[Optional[Figure], Optional[Dict[int, datetime]]]:
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None or normalized_weights is None or weekly_throughput.mean() == 0: return None, None
        complexity_factors = {'Clear and understood': 1.0, 'Somewhat understood': 1.25, 'Not really understood yet': 1.50, 'Very unclear or not understood': 2.00}
        adjusted_items = int(items_to_complete * complexity_factors.get(scope_complexity, 1.0))
        focus_factors = {'100% (only this work)': 1.0, '75% (mostly this work)': 0.75, '50% (half of this work)': 0.50, '25% (some of this work)': 0.25}
        adjusted_throughput = weekly_throughput * focus_factors.get(team_focus, 1.0)
        completion_weeks_data = []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            items_done, weeks_elapsed = 0, 0
            timeout_weeks = max(300, (adjusted_items / adjusted_throughput.mean()) * 20 if adjusted_throughput.mean() > 0 else 300)
            while items_done < adjusted_items:
                if weeks_elapsed > timeout_weeks: weeks_elapsed = -1; break
                items_done += np.random.choice(adjusted_throughput, p=normalized_weights)
                weeks_elapsed += 1
            if weeks_elapsed != -1: completion_weeks_data.append(weeks_elapsed)
        if not completion_weeks_data: return None, None
        completion_days_data = [w * 7 for w in completion_weeks_data]
        value_counts = pd.Series(completion_days_data).value_counts().sort_index()
        fig = go.Figure(data=[go.Bar(name='Simulations', x=value_counts.index, y=value_counts.values)])
        fig.update_layout(title="Forecast: Completion Date Distribution", xaxis_title=f"Days from {start_date.strftime('%d %b, %Y')} to Completion", yaxis_title="Frequency (Number of Simulations)", bargap=0.5, height=600)
        percentile_dates, percentile_colors = {}, ColorManager.get_percentile_colors(is_color_blind_mode)
        for p in Config.PERCENTILES:
            days = np.percentile(completion_days_data, p)
            percentile_dates[p] = start_date + timedelta(days=int(days))
            fig.add_vline(x=days, line_dash="dash", line_color=percentile_colors.get(p), annotation_text=f"{p}%", annotation_position="top right")
        return fig, percentile_dates

    @staticmethod
    @st.cache_data(show_spinner="Running 'When' simulations...")
    def run_when_scenario_forecast(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str) -> Optional[Dict]:
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None or normalized_weights is None or weekly_throughput.mean() == 0: return None
        median_throughput = weekly_throughput.median()
        good_weeks, bad_weeks = weekly_throughput[weekly_throughput > median_throughput], weekly_throughput[weekly_throughput <= median_throughput]
        if good_weeks.empty or bad_weeks.empty: return None
        good_completion_weeks, bad_completion_weeks = [], []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            items_done_good, weeks_elapsed_good = np.random.choice(good_weeks), 1
            while items_done_good < items_to_complete: items_done_good += np.random.choice(weekly_throughput, p=normalized_weights); weeks_elapsed_good += 1
            good_completion_weeks.append(weeks_elapsed_good * 7)
            items_done_bad, weeks_elapsed_bad = np.random.choice(bad_weeks), 1
            while items_done_bad < items_to_complete: items_done_bad += np.random.choice(weekly_throughput, p=normalized_weights); weeks_elapsed_bad += 1
            bad_completion_weeks.append(weeks_elapsed_bad * 7)
        results = {}
        for p in Config.PERCENTILES:
            results[p] = {'Good Week Start': (start_date + timedelta(days=int(np.percentile(good_completion_weeks, p)))).strftime('%d %b, %Y'), 'Bad Week Start': (start_date + timedelta(days=int(np.percentile(bad_completion_weeks, p)))).strftime('%d %b, %Y')}
        return results

    @staticmethod
    def _prepare_chart_data(df: DataFrame, columns: List[str]) -> DataFrame:
        chart_df = df.copy()
        for col in ['Completed date', 'Start date']:
            if col in chart_df.columns and not chart_df[col].dropna().empty: chart_df[f'{col.replace(" ", "_")}_formatted'] = chart_df[col].dt.strftime('%d/%m/%Y')
        for col in ['Cycle time', 'Age']:
            if col in chart_df.columns: chart_df[f'{col.replace(" ", "_")}_formatted'] = chart_df[col].astype(int).astype(str)
        chart_df['Work type'] = pd.Categorical(chart_df['Work type'], categories=ChartGenerator._order_work_types(chart_df), ordered=True)
        return chart_df

    @staticmethod
    def _order_work_types(df: DataFrame) -> List[str]:
        work_types = df['Work type'].unique()
        ordered = [wt for wt in Config.WORK_TYPE_ORDER if wt in work_types]
        ordered.extend(sorted(set(work_types) - set(Config.WORK_TYPE_ORDER)))
        return ordered

    @staticmethod
    def _add_trend_line(fig: Figure, data: DataFrame) -> None:
        if len(data) <= 2: return
        try:
            X = data['Date'].apply(lambda date: date.toordinal()).values.reshape(-1, 1)
            y = data["WIP"].values
            reg = LinearRegression().fit(X, y)
            trend_y = reg.predict(X)
            fig.add_trace(go.Scatter(x=data["Date"], y=trend_y, mode='lines', name='Trend', line=dict(color='red', dash='dash', width=2), hovertemplate=ChartConfig.TREND_LINE_HOVER))
        except Exception: pass

class StatsCalculator:
    @staticmethod
    def summary_stats(df: DataFrame) -> Dict[str, int]:
        return {'total': len(df), 'completed': df['Completed date'].notna().sum(), 'in_progress': (df['Start date'].notna() & df['Completed date'].isna()).sum()}

    @staticmethod
    @st.cache_data
    def cycle_time_stats(df: DataFrame) -> Optional[Dict[str, int]]:
        completed = df.dropna(subset=['Cycle time'])
        if completed.empty: return None
        stats = {'average': int(completed['Cycle time'].mean()), 'median': int(completed['Cycle time'].median())}
        for p in Config.PERCENTILES: stats[f'p{p}'] = int(np.percentile(completed['Cycle time'], p))
        return stats

class Dashboard:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.raw_df, self.processed_df, self.filtered_df = None, None, None
        self.status_mapping, self.selections, self.filterable_columns = {}, {}, []

    def run(self):
        st.cache_data.clear()
        st.markdown("""<style>[data-testid="stTabs"] button { font-size: 16px; font-weight: bold; padding: 10px 15px; } .guidance-expander div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; background-color: #E6E6FA; border-radius: 8px; } .guidance-expander div[data-testid="stExpander"] summary { background-color: #E6E6FA; font-size: 1.2em !important; font-style: italic !important; font-weight: bold !important; color: #31333F; }</style>""", unsafe_allow_html=True)
        with st.spinner("🔄 Processing JIRA export..."):
            loaded_df = DataProcessor.load_data(self.uploaded_file)
            if loaded_df is None: return
            self.raw_df = DataProcessor.clean_data(loaded_df)
        self.status_mapping = StatusManager.extract_status_columns(self.raw_df)
        if not self.status_mapping:
            st.error("Configuration Error: Could not find any status columns in your file.")
            return
        date_bounds_df = self._pre_process_for_sidebar()
        self.filterable_columns = self._get_filterable_columns()
        self._display_sidebar(date_bounds_df)
        self._display_getting_started_guide()
        self._display_charts()
        
    def _display_getting_started_guide(self):
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("🚀 Getting Started with Flow Metrics", expanded=False):
            st.markdown("""This dashboard helps you visualize your team's workflow using four key flow metrics, along with other chart views and statistical forecasting. The primary metrics are:

- **Cycle Time**: The total time from when work starts on an item to when it is completed.
- **Work Item Age**: The time that has passed since an item was started.
- **Work In Progress (WIP)**: The number of items that have been started but are not yet finished.
- **Throughput**: The number of work items completed in a given time period.""")
        st.markdown('</div>', unsafe_allow_html=True)

    def _get_filterable_columns(self) -> List[str]:
        # Create a set of all status names (cleaned of suffixes) and their raw column names for exclusion.
        status_names_to_exclude = set(self.status_mapping.keys())
        raw_status_cols_to_exclude = set(self.status_mapping.values())
        
        potential_filters = []
        for col in self.raw_df.columns:
            # Rule 1: Skip if it's a hardcoded exclusion like 'Key', 'Summary', etc.
            if col in Config.FILTER_EXCLUSIONS:
                continue
            
            # Rule 2: Skip if it's the global 'Work type' filter
            if col == 'Work type':
                continue

            # Rule 3: Directly check for and skip any column that starts with the '-> prefix.
            if str(col).startswith("'->"):
                continue
            
            # Rule 4: As a fallback, also check against the cleaned status names to exclude elapsed time columns.
            cleaned_col = re.sub(r'-\d+$|-DPID$', '', col).strip()
            if cleaned_col in status_names_to_exclude:
                continue

            potential_filters.append(col)
        
        # From the remaining list, only return columns that actually contain any data.
        return sorted([
            col for col in potential_filters
            if self.raw_df[col].replace('', np.nan).notna().any()
        ])

    def _pre_process_for_sidebar(self) -> DataFrame:
        status_cols = list(self.status_mapping.values())
        if not status_cols: return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})
        all_dates = []
        for col in status_cols: all_dates.extend(pd.to_datetime(self.raw_df[col].apply(DataProcessor._extract_latest_date).dropna(), errors='coerce').dropna())
        if not all_dates: return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})
        date_df = pd.DataFrame(all_dates, columns=['Date'])
        return pd.DataFrame({'Start date': [date_df['Date'].min()], 'Completed date': [date_df['Date'].max()]})
        
    def _handle_multiselect(self, key):
        # This function ensures that "All" and other options are mutually exclusive.
        prev_key = f"prev_{key}"
        
        # Initialize the 'previous' state on the first run
        if prev_key not in st.session_state:
            st.session_state[prev_key] = st.session_state[key]

        current_selection = st.session_state[key]
        previous_selection = st.session_state[prev_key]

        # Exit if there's no change to prevent unnecessary reruns
        if current_selection == previous_selection:
            return

        all_was_selected = "All" in previous_selection
        all_is_now_selected = "All" in current_selection

        # Case 1: If "All" was not selected, and the user just clicked "All"
        if not all_was_selected and all_is_now_selected:
            st.session_state[key] = ["All"]
        # Case 2: If "All" was selected, and the user just clicked another option
        elif all_was_selected and len(current_selection) > 1:
            st.session_state[key] = [s for s in current_selection if s != "All"]
        # Case 3: If the user has deselected all options
        elif not current_selection:
            st.session_state[key] = ["All"]
        
        # Update the 'previous' state for the next run
        st.session_state[prev_key] = st.session_state[key]

    def _display_sidebar(self, date_bounds_df: DataFrame):
        self._display_static_global_filters(date_bounds_df)
        self._sidebar_chart_controls()
        self._display_dynamic_filters()

    def _display_static_global_filters(self, date_bounds_df: DataFrame):
        st.sidebar.markdown("#### 📋 Global Data Filters")
        st.sidebar.caption("ℹ️ *In multi-selects, 'All' deselects others, and choosing an option deselects 'All'.*")
        work_type_key = 'work_types'
        if work_type_key not in st.session_state: st.session_state[work_type_key] = ['All']
        st.sidebar.multiselect("Work Item Type", ["All"] + ChartGenerator._order_work_types(self.raw_df), key=work_type_key, on_change=self._handle_multiselect, args=(work_type_key,))
        self.selections[work_type_key] = st.session_state[work_type_key]
        self.selections["date_range"] = st.sidebar.selectbox("Date Range", Config.DATE_RANGES, index=0, help="The selected date range does not apply to the Work Item Age chart.")
        self.selections["custom_start_date"], self.selections["custom_end_date"] = None, None
        if self.selections["date_range"] == "Custom":
            min_val, max_val = (date_bounds_df.iloc[0]['Start date'], date_bounds_df.iloc[0]['Completed date'])
            self.selections["custom_start_date"] = st.sidebar.date_input(
                "Start date", 
                value=min_val if pd.notna(min_val) else datetime.now().date(), 
                min_value=min_val, 
                max_value=max_val,
                format="DD/MM/YYYY" # FIX: Set display format
            )
            self.selections["custom_end_date"] = st.sidebar.date_input(
                "End date", 
                value=max_val if pd.notna(max_val) else datetime.now().date(), 
                min_value=min_val, 
                max_value=max_val,
                format="DD/MM/YYYY" # FIX: Set display format
            )
        self.selections["exclude_long_cycle_times"] = st.sidebar.checkbox("Exclude cycle time > 365 days", value=False)
        self.selections['color_blind_mode'] = st.sidebar.checkbox("Enable Color-Blind Friendly Mode")

    def _sidebar_chart_controls(self):
        st.sidebar.markdown("#### 📊 Chart-Specific Controls")
        with st.sidebar.expander("📈 Cycle Time & Age Percentiles", expanded=True):
            show_percentiles = st.checkbox("Show Percentile Lines", value=True)
            self.selections["percentiles"] = {f"show_{p}th": show_percentiles for p in Config.PERCENTILES}
            if show_percentiles:
                c1, c2 = st.columns(2)
                for p, col in zip(Config.PERCENTILES, [c1, c2, c1, c2]): self.selections["percentiles"][f"show_{p}th"] = col.checkbox(f"{p}th", value=True, key=f"{p}th_visible")
    
    def _display_dynamic_filters(self):
        st.sidebar.markdown("#### Dynamic Column Filters")
        if not self.filterable_columns:
            st.sidebar.info("No additional filterable columns found.")
        else:
            if 'dynamic_filters_to_show' not in st.session_state: st.session_state.dynamic_filters_to_show = []
            
            # Clean the list of available filters for display
            st.sidebar.multiselect(
                "Select Additional Filters",
                self.filterable_columns,
                key='dynamic_filters_to_show',
                help="Choose columns from your file to use as filters.",
                format_func=clean_value
            )
            
            # The session state still holds the original column names, which is correct
            for f_name in st.session_state.dynamic_filters_to_show:
                f_type = Config.FILTER_TYPE_HINTS.get(f_name, "multi")
                unique_vals = self._get_unique_values(self.raw_df[f_name], f_type)
                session_key = f"selection_{f_name}"
                
                # Clean the label of the individual filter widget for display
                display_label = clean_value(f_name)

                if f_type == "single":
                    self.selections[f_name] = st.sidebar.selectbox(
                        display_label,
                        ["All"] + unique_vals,
                        key=f"filter_{f_name}",
                        format_func=clean_value # Clean the values inside the dropdown
                    )
                else:
                    if session_key not in st.session_state: st.session_state[session_key] = ['All']
                    st.sidebar.multiselect(
                        display_label,
                        ["All"] + unique_vals,
                        key=session_key,
                        on_change=self._handle_multiselect,
                        args=(session_key,),
                        format_func=clean_value # Clean the values inside the dropdown
                    )
                    self.selections[f_name] = st.session_state[session_key]

    def _get_unique_values(self, series: pd.Series, filter_type: str) -> List[str]:
        if series.dropna().empty: return []
        
        # Replace empty strings with a more descriptive generic label "None"
        series = series.replace('', 'None').replace(np.nan, 'None')

        if filter_type == "multi":
            unique_vals = sorted(series.astype(str).str.split(',').explode().str.strip().unique())
        else:
            unique_vals = sorted(series.unique())
            
        # If "None" is in the list, move it to the front
        if "None" in unique_vals:
            unique_vals.remove("None")
            unique_vals.insert(0, "None")
            
        return unique_vals

    def _apply_all_filters(self, source_df: pd.DataFrame, apply_date_filter: bool) -> pd.DataFrame:
        if source_df is None: return pd.DataFrame()
        df = source_df.copy()
        if self.selections.get("exclude_long_cycle_times") and 'Cycle time' in df.columns: df = df[df['Cycle time'] <= 365]
        if "All" not in self.selections.get("work_types", ["All"]):
            df = df[df["Work type"].isin(self.selections["work_types"])]
            
        if 'dynamic_filters_to_show' in st.session_state:
            for f_name in st.session_state.dynamic_filters_to_show:
                selection = self.selections.get(f_name)
                if selection and "All" not in selection and f_name in df.columns:
                    
                    def check_row(val):
                        # FIX: Check for the new "None" label if the data is missing
                        if pd.isna(val) or str(val).strip() == '':
                            return 'None' in selection
                        
                        parts = [clean_value(p.strip()) for p in str(val).split(',')]
                        
                        if not any(parts) or parts == ['']:
                            return 'None' in selection
                        return any(p in selection for p in parts)
                    
                    df = df[df[f_name].apply(check_row)]

        if apply_date_filter and 'Completed date' in df.columns: 
            df = _apply_date_filter(df, 'Completed date', self.selections["date_range"], self.selections["custom_start_date"], self.selections["custom_end_date"])
        return df

    def _display_charts(self):
        tab_list = ["📈 Cycle Time", "🔄 Process Flow", "📊 Work Item Age", "⏱️ Flow Efficiency", "🔄 WIP Trend", "📊 Throughput", "🔮 Throughput Forecast"]
        
        potential_sp_cols = ['story points', 'story point estimate']
        sp_col_name = None
        for col in self.raw_df.columns:
            if col.strip().lower() in potential_sp_cols:
                if pd.to_numeric(self.raw_df[col], errors='coerce').notna().any():
                    sp_col_name = col
                    break
        
        if sp_col_name:
            tab_list.append("📈 Story Point Analysis")
            
        tabs = st.tabs(tab_list)
        tab_map = {"📈 Cycle Time": self._display_cycle_time_charts, "🔄 Process Flow": self._display_cfd_chart, "📊 Work Item Age": self._display_work_item_age_chart, "⏱️ Flow Efficiency": self._display_flow_efficiency_chart, "🔄 WIP Trend": self._display_wip_chart, "📊 Throughput": self._display_throughput_chart, "🔮 Throughput Forecast": self._display_forecast_charts, "📈 Story Point Analysis": self._display_story_point_chart}
        for i, tab_title in enumerate(tab_list):
            with tabs[i]: 
                if tab_title in tab_map:
                    tab_map[tab_title]()

    def _display_flow_efficiency_chart(self):
        st.header("Flow Efficiency Analysis")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
                - **What it is:** Flow Efficiency is the percentage of time that work items spend in 'active' work states versus 'wait' states...
                - **Formula:** `Flow Efficiency = (Total Active Time / Total Cycle Time) * 100`
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        start_status, end_status = self.selections.get('start_status'), self.selections.get('completed_status')
        if not start_status or start_status == "None" or not end_status or end_status == "None":
            st.info("To view this chart, first configure your cycle time statuses on the '📈 Cycle Time' tab.")
            return

        st.info(f"""
        - **Calculation based on:** Cycle Time from `{clean_value(start_status)}` to `{clean_value(end_status)}` (set on the Cycle Time tab).
        - **Wait/Queue Statuses:** Please select the statuses below that represent idle/waiting time.
        - **Note:** To provide a more holistic view of process health, this Flow Efficiency calculation includes **all work items**, both completed and in-progress. The 'Cycle Time' metric on this page may differ from the scatter plot, as it is derived from the sum of time spent in each status for this wider set of items.
        """)

        all_statuses_ordered = list(self.status_mapping.keys())
        all_statuses_in_cycle = []
        try:
            start_index = all_statuses_ordered.index(start_status)
            end_index = all_statuses_ordered.index(end_status)
            if start_index < end_index:
                all_statuses_in_cycle = all_statuses_ordered[start_index:end_index]
        except (ValueError, TypeError):
            pass

        default_wait = [s for s in all_statuses_in_cycle if clean_value(s).lower() in ['to do', 'ready for dev', 'ready for test']]
        wait_statuses = st.multiselect(
            "Select Wait/Queue Statuses", 
            all_statuses_in_cycle, 
            default=default_wait, 
            help="Select statuses that represent idle/waiting time.",
            format_func=clean_value
        )
        
        if not wait_statuses:
            st.warning("Please select one or more 'Wait/Queue' statuses to calculate Flow Efficiency.")
            return
        
        source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        
        efficiency_df = DataProcessor.calculate_flow_efficiency(source_df, all_statuses_in_cycle, wait_statuses)
        
        if efficiency_df.empty or efficiency_df['Cycle time'].isnull().all() or (efficiency_df['Cycle time'] <= 0).all():
            st.warning("No data available to calculate Flow Efficiency with the current filters.")
            return

        p85_tab, avg_tab = st.tabs(["85th Percentile", "Average"])

        with p85_tab:
            st.markdown("This view shows the **85th percentile** values...")
            p85_cycle_time = efficiency_df['Cycle time'].quantile(0.85)
            p85_wait_time = efficiency_df['Wait Time Days'].quantile(0.85)
            
            p85_active_time = p85_cycle_time - p85_wait_time
            p85_active_time = max(0, p85_active_time) 
            p85_efficiency = (p85_active_time / p85_cycle_time) * 100 if p85_cycle_time > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("85p Active Time", f"{p85_active_time:.2f} Days")
            c2.metric("85p Wait Time", f"{p85_wait_time:.2f} Days")
            c3.metric("85p Cycle Time", f"{p85_cycle_time:.2f} Days")
            c4.metric("85p Flow Efficiency", f"{p85_efficiency:.1f}%")

        with avg_tab:
            st.markdown("This view shows the **average** (mean) values for your flow data.")
            avg_cycle_time = efficiency_df['Cycle time'].mean()
            avg_wait_time = efficiency_df['Wait Time Days'].mean()
            avg_active_time = efficiency_df['Active Time Days'].mean()
            avg_efficiency = (avg_active_time / avg_cycle_time) * 100 if avg_cycle_time > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average Active Time", f"{avg_active_time:.2f} Days")
            c2.metric("Average Wait Time", f"{avg_wait_time:.2f} Days")
            c3.metric("Average Cycle Time", f"{avg_cycle_time:.2f} Days")
            c4.metric("Average Flow Efficiency", f"{avg_efficiency:.1f}%")

        st.markdown("---")
        st.subheader("Flow Efficiency Distribution")
        st.markdown("This chart shows the distribution of Flow Efficiency scores for all items analyzed. The metric tiles above are derived from this data.")
        
        chart = ChartGenerator.create_flow_efficiency_histogram(efficiency_df)
        if chart: 
            st.plotly_chart(chart, use_container_width=True)
            with st.expander("Where is the time going? (Time in Status)", expanded=True):
                st.markdown("""
                These charts show how long items spend in each status...
                """)
                
                p85_tab_exp, avg_tab_exp = st.tabs(["85th Percentile Time", "Average Time"])
                status_cols = list(self.status_mapping.values())

                with p85_tab_exp:
                    st.markdown("This chart shows the **85th percentile** time spent in each status...")
                    p85_chart, _ = ChartGenerator.create_85th_time_in_status_chart(self.filtered_df, status_cols)
                    if p85_chart:
                        st.plotly_chart(p85_chart, use_container_width=True)
                with avg_tab_exp:
                    st.markdown("This chart shows the **average** time items spend in each status.")
                    time_in_status_chart, _ = ChartGenerator.create_time_in_status_chart(self.filtered_df, status_cols)
                    if time_in_status_chart:
                        st.plotly_chart(time_in_status_chart, use_container_width=True)
                    else:
                        st.warning("Not enough data to display the Average Time in Status chart.")
        else: 
            st.warning("⚠️ Could not generate the Flow Efficiency distribution chart.")

    def _display_cfd_chart(self):
        """Displays the Cumulative Flow Diagram and its controls."""
        st.header("Process Stability & Flow")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
             st.markdown("""
                - **What it is:** This chart shows the cumulative number of work items in each stage of your workflow over time. It helps visualize the flow of work and identify bottlenecks.
                - **How to read it:** Each colored band represents a stage in your workflow. The vertical distance between the lines shows the number of items in that stage on a given day (the Work In Progress for that stage). The horizontal distance represents the approximate cycle time for that stage.
                - **What to look for:**
                    - **Widening Bands:** If a colored band is getting wider over time, it indicates that more work is arriving in that stage than is leaving it. This is a classic sign of a **bottleneck**.
                    - **Flat Bands:** If all bands are flat, it means no work is being completed.
                    - **Parallel Bands:** If the top and bottom lines of the chart are moving in parallel, it generally indicates a **stable flow** where work is arriving and leaving at a similar rate.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        all_statuses = list(self.status_mapping.keys())
        self.selections['cfd_statuses'] = st.multiselect(
            "Select workflow statuses in order",
            options=all_statuses,
            default=all_statuses,
            help="Select the statuses you want to appear in the chart. The order of selection determines the stacking order."
        )

        if not self.selections['cfd_statuses']:
            st.info("ℹ️ Please select at least one status to generate the Cumulative Flow Diagram.")
            return

        source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)

        chart = ChartGenerator.create_cumulative_flow_diagram(
            source_df,
            self.selections['cfd_statuses'],
            self.status_mapping,
            self.selections['date_range'],
            self.selections['custom_start_date'],
            self.selections['custom_end_date'],
            self.selections['color_blind_mode']
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ No data available to generate the Cumulative Flow Diagram for the selected criteria.")

    def _display_cycle_time_charts(self):
        """Displays the Cycle Time charts and statistics."""
        st.header("Cycle Time Analysis")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about Cycle Time", icon="🎓"):
            st.markdown("""
            - **What it is:** Cycle time is how long it takes to complete a work item from the moment work begins until it is delivered. These charts help you visualize the consistency and predictability of your team's delivery over time.
            - **Why it matters:** A stable and predictable cycle time allows for more reliable forecasting. Consistently high or increasing cycle times can indicate systemic problems like bottlenecks, work items being too large, or external dependencies.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)

        with col1:
            self.selections["start_status"] = st.selectbox(
                "Define your 'Start' status for Cycle Time calculation", 
                status_options, 
                key="cycle_time_start",
                format_func=clean_value
            )
        with col2:
            self.selections["completed_status"] = st.selectbox(
                "Define your 'Done' status for Cycle Time calculation", 
                status_options, 
                key="cycle_time_end",
                format_func=clean_value
            )

        self.selections["start_col"] = self.status_mapping.get(self.selections["start_status"])
        self.selections["completed_col"] = self.status_mapping.get(self.selections["completed_status"])

        if not self.selections["start_col"] or not self.selections["completed_col"]:
            st.info("To see your charts, please choose a 'Starting Status' and a 'Done Status' from the dropdowns above.")
            self.filtered_df = pd.DataFrame() 
            return

        is_valid, error_msg = StatusManager.validate_status_order(self.raw_df, self.selections["start_col"], self.selections["completed_col"])
        if not is_valid:
            st.error(error_msg)
            return

        # --- FIX: Calculate Summary Stats from a broader dataset FIRST ---
        # Apply global filters (Work Type, etc.) to the raw data
        stats_source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        
        # Calculate stats based on the selected "Done" status
        total_items = len(stats_source_df)
        completed_items = stats_source_df[self.selections["completed_col"]].apply(lambda x: pd.notna(DataProcessor._extract_latest_date(x))).sum()
        in_progress_items = total_items - completed_items
        
        m1, m2, m3 = st.columns(3)
        m1.metric("📊 Total Items in Filter", total_items)
        m2.metric("✅ Completed Items", completed_items)
        m3.metric("🔄 Still In Progress", in_progress_items)
        # --- END FIX ---

        self.processed_df = DataProcessor.process_dates(self.raw_df, self.selections["start_col"], self.selections["completed_col"])
        if self.processed_df is None:
             st.warning("Could not calculate Cycle Time with the selected statuses. Please check your selections.")
             return

        self.filtered_df = self._apply_all_filters(self.processed_df, apply_date_filter=True)
        
        if self.filtered_df.empty or self.filtered_df['Completed date'].isna().all():
            st.warning("No completed items found for the selected criteria. Unable to display Cycle Time charts.")
            return
        
        ct_tabs = st.tabs(["Scatter Plot", "Bubble Chart", "Box Plot", "Distribution (Histogram)", "Time in Status"])
        with ct_tabs[0]:
            with st.expander("How to Read This Chart", icon="🎓"):
                st.markdown("""
                - **How to read it:** Each dot is a completed work item...
                """)
            st.markdown("ℹ️ *A small amount of random vertical 'jitter' has been added to separate overlapping points.*")
            
            plottable_df = self.filtered_df.dropna(subset=['Start date', 'Completed date', 'Cycle time'])
            unplotted_df = self.filtered_df[self.filtered_df['Start date'].isna() & self.filtered_df['Completed date'].notna()]

            chart = ChartGenerator.create_cycle_time_chart(plottable_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            
            if chart:
                st.plotly_chart(chart, use_container_width=True)
                if not unplotted_df.empty:
                    with st.expander(f"Sanity Check: {len(unplotted_df)} completed item(s) not plotted"):
                        st.markdown(f"""
                        The following items are included in the **'Completed Items'** metric above but are **not shown as dots on the chart**.
                        This is because their cycle time cannot be calculated as they are missing a date in the selected **'Start' status**: `{clean_value(self.selections["start_status"])}`.
                        This can happen if items skip the start of your workflow.
                        """)
                        display_cols = ['Key', 'Summary', 'Status', 'Work type', 'Completed date']
                        existing_display_cols = [col for col in display_cols if col in unplotted_df.columns]
                        st.dataframe(unplotted_df[existing_display_cols], use_container_width=True, hide_index=True)
            else: 
                st.warning("No completed items in the selected date range could be found to display on this chart.")

        with ct_tabs[1]:
            with st.expander("How to Read This Chart", icon="🎓"):
                st.markdown("""
                - **How to Read It:** Each bubble's position shows the cycle time and completion date...
                """)
            st.markdown("ℹ️ *Bubbles represent one or more items completed on the same day with the same cycle time.*")
            chart = ChartGenerator.create_cycle_time_bubble_chart(self.filtered_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[2]:
            with st.expander("How to Read This Chart", icon="🎓"):
                st.markdown("""
                - **How to Read It:** This statistical chart summarizes your cycle time for each period...
                """)
            self.selections["box_plot_interval"] = st.selectbox("Group Box Plot by", ["Weekly", "Monthly"], index=0)
            chart = ChartGenerator.create_cycle_time_box_plot(self.filtered_df, self.selections["box_plot_interval"], self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[3]:
            chart = ChartGenerator.create_cycle_time_histogram(self.filtered_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[4]:
            p85_tab, avg_tab = st.tabs(["85th Percentile Time", "Average Time"])
            status_cols = list(self.status_mapping.values())

            with p85_tab:
                st.markdown("This chart shows the **85th percentile** time spent in each status...")
                p85_chart, p85_data = ChartGenerator.create_85th_time_in_status_chart(self.filtered_df, status_cols)
                if p85_chart:
                    st.plotly_chart(p85_chart, use_container_width=True)
                    if p85_data is not None and not p85_data.empty:
                        st.divider()
                        num_statuses = len(p85_data)
                        cols = st.columns(min(num_statuses, 5), gap="small")
                        for i, row in p85_data.iterrows():
                            col = cols[i % 5]
                            col.metric(label=row['Status'], value=f"{int(row['85th Percentile Time (Days)'])} days")
                else:
                    st.warning("Not enough data to display the 85th Percentile Time in Status chart.")

            with avg_tab:
                st.markdown("This chart shows the **average** time items spend in each status column of your raw data export.")
                chart, chart_data = ChartGenerator.create_time_in_status_chart(self.filtered_df, status_cols)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    if chart_data is not None and not chart_data.empty:
                        st.divider()
                        num_statuses = len(chart_data)
                        cols = st.columns(min(num_statuses, 5), gap="small")
                        for i, row in chart_data.iterrows():
                            col = cols[i % 5]
                            col.metric(label=row['Status'], value=f"{int(row['Average Time (Days)'])} days")
                else:
                    st.warning("Not enough data was found for the selected statuses to calculate the average time spent in each.")

    def _display_story_point_chart(self):
        """Displays the Story Point Correlation chart and its controls."""
        st.header("Story Point Analysis")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
                - **What it is:** This chart plots the cycle time of completed items against their story point estimates. It's a tool to check the correlation between estimates and actual time taken.
                - **How to read it:** Each dot is a work item. The horizontal position is its story point value, and the vertical position is its cycle time.
                - **What to look for:**
                    - **No Correlation (Ideal for Flow):** If there is little to no relationship between story points and cycle time (dots are scattered randomly across different point values), it can indicate that your team is effectively "right-sizing" work. This means you are breaking down work into small, similarly-sized pieces, regardless of the initial estimate. In a mature flow-based system, this is often a desirable outcome.
                    - **Positive Correlation:** If cycle time tends to increase as story points increase, it means your estimates are somewhat predictive of effort. However, a wide vertical spread for any given story point value still indicates high variability.
                    - **High Variability within a Story Point:** If a single story point value (e.g., 5 points) has a very wide range of cycle times (from 5 to 50 days), it highlights that story points are not a reliable predictor of completion time for your team.
                    - **Unreliable Estimates:** If different story point estimates (e.g., 2, 3, and 5 points) all show a similarly wide range of cycle times, it suggests the assigned estimate has little influence on the actual completion time. This indicates that story points are not a reliable predictor for forecasting when work will be done.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)

        with col1:
            self.selections["sp_start_status"] = st.selectbox("Starting Status", status_options, key="sp_start")
        with col2:
            self.selections["sp_end_status"] = st.selectbox("Done Status", status_options, key="sp_end", help="Please select the status that represents your Definition of Done.")

        self.selections["sp_start_col"] = self.status_mapping.get(self.selections["sp_start_status"])
        self.selections["sp_end_col"] = self.status_mapping.get(self.selections["sp_end_status"])

        if not self.selections["sp_start_col"] or not self.selections["sp_end_col"]:
            st.info("ℹ️ Please select a 'Starting Status' and 'Done Status' above to generate the chart.")
            return

        sp_processed_df = DataProcessor.process_dates(self.raw_df, self.selections["sp_start_col"], self.selections["sp_end_col"])
        
        if sp_processed_df is None:
            st.warning("Could not process dates with the selected status columns. Please check your data or selections.")
            return

        sp_filtered_df = self._apply_all_filters(sp_processed_df, apply_date_filter=True)

        chart = ChartGenerator.create_story_point_chart(sp_filtered_df, self.selections['color_blind_mode'])
        if chart:
            chart_col, _ = st.columns([0.75, 0.25])
            with chart_col:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ No completed items with story points found for the selected criteria.")

    def _display_work_item_age_chart(self):
        """Displays the Work Item Age chart and its controls."""
        st.header("Work Item Age Analysis")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
            - **What it is:** This chart shows all the items that are currently in progress, their current status (column), and how long they have been in progress (their Age). Age is the "running clock" for items that have not yet finished.
            - **How to read it:** Each dot is a work item that has started but not yet finished. Its vertical position shows its current age in days.
            - **What to look for:**
                - **The oldest items first.** The most important question in a Daily Stand-up is "what's the oldest thing we are working on, and what are we doing to get it moving?" Items at the top of the chart are your oldest and represent the most risk.
                - **Items nearing or crossing percentile lines.** The percentile lines are taken from your historical Cycle Time data. If an item's age is approaching the 85th percentile, it is at high risk of taking longer than 85% of all your previous items. This is a crucial signal to the team to intervene by swarming on the item or breaking it down.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        status_options = ["None"] + list(self.status_mapping.keys())
        sensible_done_options = [s for s in status_options if s.lower() == 'done']
        default_done_index = status_options.index(sensible_done_options[0]) if sensible_done_options else len(status_options) - 1

        col1, col2, col3 = st.columns(3)
        with col1:
            self.selections["age_start_status"] = st.selectbox("Start Status for Age Calculation", status_options, key="age_start", index=1 if len(status_options) > 1 else 0, help="Defines the start of the X-axis and the point from which item age is calculated.")
        with col2:
            try:
                default_end_index = status_options.index("In Testing") if "In Testing" in status_options else len(status_options) - 2
            except ValueError:
                default_end_index = len(status_options) - 2 if len(status_options) > 2 else 0
            self.selections["age_end_status"] = st.selectbox("End Status for Axis", status_options, key="age_end", index=max(0, default_end_index), help="Defines the end of the X-axis.")
        with col3:
            self.selections["age_true_final_status"] = st.selectbox("Select the true 'Done' status", status_options, key="age_final", index=default_done_index, help="Select the status that marks an item as completely finished for your workflow.")

        self.selections["age_start_col"] = self.status_mapping.get(self.selections["age_start_status"])
        self.selections["age_end_col"] = self.status_mapping.get(self.selections["age_end_status"])
        self.selections["age_true_final_col"] = self.status_mapping.get(self.selections["age_true_final_status"])
        
        if not all([self.selections["age_start_col"], self.selections["age_end_col"], self.selections["age_true_final_col"]]):
            st.info("To see the chart, please select a 'Start Status for Age Calculation', an 'End Status for Axis', and a true 'Done' status from the controls above.")
            return

        try:
            raw_status_keys = list(self.status_mapping.keys())
            start_idx = raw_status_keys.index(self.selections["age_start_status"])
            end_idx = raw_status_keys.index(self.selections["age_end_status"])
            if start_idx >= end_idx:
                st.error("The 'Start Status for Age Calculation' must come before the 'End Status for Axis' in your workflow.")
                return
            status_order = raw_status_keys[start_idx : end_idx + 1]
        except (ValueError, IndexError):
            st.error("Could not determine status order for the chart axis. Check your axis status selections.")
            return

        all_in_progress_df = self.raw_df[
            self.raw_df[self.selections["age_true_final_col"]].apply(lambda x: pd.isna(DataProcessor._extract_latest_date(x)))
        ].copy()
        
        filtered_in_progress_df = self._apply_all_filters(all_in_progress_df, apply_date_filter=False)
        wip_on_axis_df = filtered_in_progress_df[filtered_in_progress_df['Status'].isin(status_order)].copy()
        wip_counts_by_status = wip_on_axis_df['Status'].value_counts().to_dict()
        wip_on_axis_df['Start date'] = wip_on_axis_df[self.selections["age_start_col"]].apply(DataProcessor._extract_latest_date)
        plot_df = wip_on_axis_df[wip_on_axis_df['Start date'].notna()].copy()
        unplotted_df = wip_on_axis_df[wip_on_axis_df['Start date'].isna()].copy()

        cycle_stats = StatsCalculator.cycle_time_stats(self.filtered_df) if self.filtered_df is not None and not self.filtered_df.empty else None

        chart = ChartGenerator.create_work_item_age_chart(
            plot_df,
            wip_counts_by_status,
            status_order,
            cycle_stats or {},
            self.selections["percentiles"],
            self.selections['color_blind_mode']
        )
        
        if chart:
            st.info("""
            - WIP counts at the top show all in-progress items currently in that status based on your filters.
            - Dots on the chart represent only those items that have passed through the selected 'Start Status for Age Calculation'.
            - The percentile lines are based on the cycle time of completed items (from the "Cycle Time" tab) to help you gauge if aging items are approaching your typical completion times.

            *N.B. You may see a discrepancy between total WIP count and the number of dots shown. This is normal behaviour and explained in the Sanity Check expander below the chart.*
            """)
            st.plotly_chart(chart, use_container_width=True)

            if not unplotted_df.empty:
                with st.expander(f"Sanity Check: {len(unplotted_df)} item(s) counted in WIP but not plotted"):
                    st.markdown(f"""
                    The following items are included in the **WIP counts** at the top of the chart columns but are **not shown as dots**.
                    This is because their age cannot be calculated as they have **not yet passed through the selected 'Start Status'**: `{self.selections["age_start_status"]}`.
                    This can happen if items are created directly in a later workflow status, or if items do not pass through some statuses when being updated in Jira.
                    """)
                    
                    display_df = unplotted_df.copy()
                    display_df['Status'] = pd.Categorical(display_df['Status'], categories=status_order, ordered=True)
                    display_df.sort_values('Status', inplace=True)
                    
                    def get_status_timestamp(row):
                        raw_status_col = self.status_mapping.get(row['Status'])
                        if raw_status_col and raw_status_col in row:
                            latest_date = DataProcessor._extract_latest_date(row[raw_status_col])
                            if pd.notna(latest_date):
                                return latest_date.strftime('%d/%m/%Y %H:%M')
                        return None

                    display_df['Status Timestamp'] = display_df.apply(get_status_timestamp, axis=1)
                    
                    display_cols = ['Key', 'Summary', 'Status', 'Status Timestamp']
                    existing_display_cols = [col for col in display_cols if col in display_df.columns]
                    st.dataframe(display_df[existing_display_cols], use_container_width=True, hide_index=True)
        else: 
            st.warning("No work items currently in progress were found based on your status selections and filters.")

    def _display_wip_chart(self):
        st.header("Work In Progress (WIP) Trend")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""- **What it is:** This chart shows the number of work items that are in progress at any given point in time.\n- **How to read it:** The line shows the total count of in-progress items for each day in the selected period.\n- **What to look for:**\n    - **Rising WIP:** A consistent upward trend in WIP is a warning sign. According to Little's Law, if your throughput remains constant, a rising WIP will directly lead to longer cycle times.\n    - **Large Spikes and Drops:** Significant fluctuations in WIP can indicate that work is being started in large batches, which can strain the system. Aim for a stable, limited WIP.\n    - **Relationship to Cycle Time:** Compare the WIP chart with your Cycle Time scatterplot. Periods of high WIP will often correspond to periods of longer cycle times.""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)
        with col1: self.selections["wip_start_status"] = st.selectbox("WIP Start Status", status_options, key="wip_start")
        with col2: self.selections["wip_done_status"] = st.selectbox("WIP End Status", status_options, key="wip_end")
        self.selections["wip_start_col"] = self.status_mapping.get(self.selections["wip_start_status"])
        self.selections["wip_done_col"] = self.status_mapping.get(self.selections["wip_done_status"])

        if not self.selections["wip_start_col"] or not self.selections["wip_done_col"]:
            st.info("ℹ️ Please select a Start and End Status above to generate the WIP chart.")
            return
            
        wip_processed_df = DataProcessor.process_dates(self.raw_df, self.selections["wip_start_col"], self.selections["wip_done_col"])
        source_df = self._apply_all_filters(wip_processed_df, apply_date_filter=False)
        chart = ChartGenerator.create_wip_chart(source_df, self.selections['date_range'], self.selections['custom_start_date'], self.selections['custom_end_date'])
        if chart: st.plotly_chart(chart, use_container_width=True)
        else: st.warning("⚠️ No items with start dates for WIP chart.")

    def _display_throughput_chart(self):
        st.header("Throughput")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""- **What it is:** This chart shows the number of work items completed per unit of time (e.g., week, fortnight). Throughput is a measure of the team's delivery rate.\n- **How to read it:** Each bar represents a time period, and its height shows the number of items that were completed in that period.\n- **What to look for:**\n    - **Consistency:** A relatively consistent throughput over time indicates a stable and predictable process.\n    - **Variability:** High variability (large spikes and drops) can suggest that work items are not "right-sized" or that the team is being affected by outside interruptions or dependencies.\n    - **Zero Throughput:** Any period with zero throughput is worth investigating.""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1: self.selections["throughput_interval"] = st.selectbox("Interval", Config.THROUGHPUT_INTERVALS, key="throughput_interval_selector")
        with col2:
            status_options = ["None"] + list(self.status_mapping.keys())
            if 'throughput_status_key' not in st.session_state: st.session_state.throughput_status_key = status_options[-1] if len(status_options) > 1 else "None"
            self.selections["throughput_status"] = st.selectbox("Choose the throughput status", status_options, key="throughput_status_key")
        self.selections['sprint_anchor_date'] = None
        if self.selections["throughput_interval"] == 'Fortnightly':
            with col3: self.selections['sprint_anchor_date'] = st.date_input("Sprint End Date", value=datetime.now(), help="Select the last day of any Sprint in your team's cadence.")
        self.selections['throughput_status_col'] = self.status_mapping.get(self.selections["throughput_status"])

        source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        max_date_df = self.raw_df.copy()
        overall_max_date = None
        if self.selections['throughput_status_col']:
            max_date_df['Throughput Date'] = max_date_df[self.selections['throughput_status_col']].apply(DataProcessor._extract_latest_date)
            max_date_df.dropna(subset=['Throughput Date'], inplace=True)
            if not max_date_df.empty: overall_max_date = max_date_df['Throughput Date'].max()
        chart, _ = ChartGenerator.create_throughput_chart(source_df, self.selections["throughput_interval"], self.selections['throughput_status_col'], self.selections['date_range'], self.selections['custom_start_date'], self.selections['custom_end_date'], sprint_anchor_date=self.selections.get('sprint_anchor_date'), overall_max_date=overall_max_date)
        if chart: st.plotly_chart(chart, use_container_width=True)
        else: st.warning("⚠️ No items with the selected throughput status for this chart.")

    def _display_forecast_charts(self):
        st.header("Throughput Forecasting")
        st.markdown('<div class="guidance-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about forecasting", icon="🎓"):
            st.markdown("""- **What it is:** These charts use a **Monte Carlo simulation** to forecast future outcomes based on your team's historical throughput data. Instead of giving a single, misleading date, it provides a range of outcomes and their probabilities.\n- **How to read it:** The charts run thousands of simulations of your future work to generate a range of possible outcomes. For example, a result might say "There is an 85% chance of completing 12 or more items in the next two weeks."\n- **Why use Monte Carlo?** Traditional forecasting often uses simple averages, which can be misleading and hide risk (the "Flaw of Averages"). A Monte Carlo simulation is a more robust statistical method that accounts for the variability in your past performance.\n- **A Note on "Right-Sizing":** Forecasts are most reliable when the work items are "right-sized." This means each item should be broken down into the smallest possible chunk that still delivers value and can be completed within your team's Service Level Expectation (SLE).""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        throughput_status = self.selections.get('throughput_status')
        if not throughput_status or throughput_status == "None":
            st.info("To run a forecast, first go to the 'Throughput' tab and choose the status that represents work being completed.")
            return
        st.info(f"Uses historical throughput data to run Monte Carlo simulations. The forecast is based on items reaching the **'{throughput_status}'** status.")
        forecast_source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        throughput_status_col = self.selections.get('throughput_status_col')
        weekly_throughput, _ = ChartGenerator._get_recent_weekly_throughput(forecast_source_df, throughput_status_col)
        with st.expander("Data Stability Check"):
            if weekly_throughput is None or len(weekly_throughput) < 4:
                st.warning("Not enough historical data for a stability check. At least 4 weeks of data are needed.")
            else: self._display_stability_check(weekly_throughput)
        forecast_tabs = st.tabs(["**How Many** (by date)", "**When** (by # of items)"])
        with forecast_tabs[0]:
            st.subheader("How many items can we complete by a certain date?")
            col1, col2 = st.columns([1, 1])
            with col1: self.selections["forecast_range"] = st.selectbox("Forecast Timeframe", Config.FORECAST_DATE_RANGES, index=0, key="how_many_timeframe")
            self.selections["forecast_custom_date"] = None
            if self.selections["forecast_range"] == "Custom":
                with col2: self.selections["forecast_custom_date"] = st.date_input("Forecast End Date", min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=30), key="how_many_custom_date")
            
            if self.selections.get("forecast_range") == "Next 30 days": forecast_days = 30
            elif self.selections.get("forecast_range") == "Next 60 days": forecast_days = 60
            elif self.selections.get("forecast_range") == "Next 90 days": forecast_days = 90
            elif self.selections.get("forecast_range") == "Custom" and self.selections.get("forecast_custom_date"): forecast_days = max(1, (self.selections.get("forecast_custom_date") - datetime.now().date()).days)
            else: forecast_days = 30
            chart = ChartGenerator.create_how_many_forecast_chart(forecast_source_df, forecast_days, throughput_status_col, self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("⚠️ Insufficient historical data for forecasting.")
        with forecast_tabs[1]:
            st.subheader("When can we expect to complete a given number of items?")
            col1, col2, col3, col4 = st.columns(4)
            with col1: items_to_complete = st.number_input("Number of items to forecast:", min_value=1, value=20, step=1, key="when_forecast_items")
            with col2: scope_complexity = st.selectbox("Scope Complexity", ['Clear and understood', 'Somewhat understood', 'Not really understood yet', 'Very unclear or not understood'], key="scope_complexity")
            with col3: team_focus = st.selectbox("Team Focus", ['100% (only this work)', '75% (mostly this work)', '50% (half of this work)', '25% (some of this work)'], key="team_focus")
            with col4: forecast_start_date = st.date_input("Forecast start date", value=datetime.now().date(), key="when_forecast_start")
            
            chart, stats = ChartGenerator.create_when_forecast_chart(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col, scope_complexity, team_focus, self.selections['color_blind_mode'])
            if stats:
                st.markdown("""<style>.forecast-box { padding: 10px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px; } .forecast-label { font-size: 1.1em; font-weight: bold; } .forecast-value { font-size: 2em; font-weight: bold; }</style>""", unsafe_allow_html=True)
                box_colors, text_color = ColorManager.get_forecast_box_colors(self.selections['color_blind_mode']), "#212529"
                cols = st.columns(len(stats))
                for i, (p, date_val) in enumerate(stats.items()):
                    with cols[i]: st.markdown(f"""<div class="forecast-box" style="background-color: {box_colors.get(p, '#e9ecef')}; color: {text_color};"><div class="forecast-label">{p}% Likelihood</div><div class="forecast-value">{date_val.strftime("%d %b, %Y")}</div></div>""", unsafe_allow_html=True)

            with st.expander("🤔 What does it mean if the percentile dates are the same?"):
                st.markdown("""
                This is not an error. It's a positive sign that your forecast is **reliable** because it's based on very consistent historical performance.

                The forecast is calculated by simulating thousands of future scenarios based on your past weekly completion rates (throughput). If your throughput is very stable, the vast majority of simulations will finish in the **exact same number of weeks**.

                When more than 85% of the simulations give the same result (e.g., "3 weeks"), the 50th, 70th, and 85th percentiles will all naturally calculate to the same final date.

                **In short: the closer the dates are, the more predictable your process is, and the more confidence you can have in the forecast.**
                """)

            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("⚠️ Insufficient historical data for forecasting.")
            with st.expander("🔍 Explore Forecast Scenarios"):
                st.markdown("This section explores how your completion date changes depending on whether your first week is 'good' (above median) or 'bad' (at or below median).")
                scenario_stats = ChartGenerator.run_when_scenario_forecast(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col)
                if scenario_stats and 'median' in scenario_stats:
                    st.subheader(f"Scenario 1: Good First Week (> {int(scenario_stats['median'])} items)")
                    cols = st.columns(len(scenario_stats['good_week']))
                    for i, (p, date_val) in enumerate(scenario_stats['good_week'].items()): cols[i].metric(label=f"{p}% Likelihood", value=date_val.strftime("%d %b, %Y"))
                    st.subheader(f"Scenario 2: Bad First Week (≤ {int(scenario_stats['median'])} items)")
                    cols = st.columns(len(scenario_stats['bad_week']))
                    for i, (p, date_val) in enumerate(scenario_stats['bad_week'].items()): cols[i].metric(label=f"{p}% Likelihood", value=date_val.strftime("%d %b, %Y"))
                else: st.info("Not enough data to run scenario analysis.")

    def _display_stability_check(self, throughput_data: pd.Series):
        shuffled_data = throughput_data.sample(frac=1).reset_index(drop=True)
        split_point = len(shuffled_data) // 2
        group1, group2 = shuffled_data.iloc[:split_point], shuffled_data.iloc[split_point:]
        avg1, avg2 = group1.mean(), group2.mean()
        stability_score = 0 if avg1 == 0 and avg2 == 0 else abs(avg1 - avg2) / ((avg1 + avg2) / 2) * 100
        if stability_score < 15: score_category, message = "Stable", f"✅ **Stable Data ({stability_score:.1f}% variation):** Your historical data is consistent and well-suited for forecasting."
        elif stability_score < 30: score_category, message = "Some Variability", f"⚠️ **Some Variability ({stability_score:.1f}% variation):** Your data shows some inconsistencies. The forecast is still useful but should be considered with this in mind."
        else: score_category, message = "High Variability", f"🚨 **High Variability ({stability_score:.1f}% variation):** Your process appears unstable. Forecasts based on this data may be unreliable and should be used with caution."
        st.markdown(f'<p style="color:{ColorManager.STABILITY_SCORE_COLORS[score_category]}; font-weight: bold;">{message}</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: st.markdown("##### Group 1 (Random Half)"); st.metric("Average Throughput", f"{avg1:.2f}"); st.metric("Median Throughput", f"{group1.median():.2f}")
        with col2: st.markdown("##### Group 2 (Random Half)"); st.metric("Average Throughput", f"{avg2:.2f}"); st.metric("Median Throughput", f"{group2.median():.2f}")
        if score_category == "High Variability": st.markdown("""---
            **Potential Causes of High Variability:**\n- **Work Item Size:** A mix of very large and very small items.\n- **Team Availability:** Holidays or sick leave impacting some weeks.\n- **Batching of Work:** Closing many items at once instead of a steady flow.\n- **External Blockers:** Periods of waiting for other teams or information.\n- **Process or Team Changes:** Recent changes to your team's structure or way of working.""")

def display_welcome_message():
    st.markdown("### 👋 Welcome to the Flow Metrics Dashboard")
    def _create_inline_link_with_logo(text: str, logo_path: str, url: str) -> str:
        if not os.path.exists(logo_path): return f'<a href="{url}" target="_blank">**{text}**</a>'
        try:
            with open(logo_path, "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
            return f'<a href="{url}" target="_blank" style="text-decoration: none; color: inherit; font-weight: bold;"><img src="data:image/png;base64,{logo_b64}" style="height: 1.1em; vertical-align: -0.2em; margin-right: 5px;">{text}</a>'
        except Exception: return f'<a href="{url}" target="_blank">**{text}**</a>'
    pro_link = _create_inline_link_with_logo("Status Time Reports", "Status time pro icon.png", "https://marketplace.atlassian.com/apps/1221826/status-time-reports-time-in-status")
    free_link = _create_inline_link_with_logo("Status Time Reports Free", "Status time free icon.png", "https://marketplace.atlassian.com/apps/1222051/status-time-reports-free-time-in-status?hosting=cloud&tab=overview")
    st.markdown(f"A dynamic set of flow metrics charts and forecasting built to analyze data exported from the Jira plugins {pro_link} & {free_link}.", unsafe_allow_html=True)
    with st.expander("How to export your data from Jira", expanded=True):
        st.markdown("""
        #### Export Settings
        1. **Choose your data**: In the plugin, select the projects, filters, and work item types you want to analyze.
        2. **Order your Status columns**: For the most accurate charts, it's best to order the Status columns in the export settings to match your team's workflow (e.g., `Backlog` -> `To Do` -> `In Progress` -> `Done`).
        3. **Select 'Show entry dates'**: You **must** select this from the Report List dropdown. This creates the `-> Status` columns that this dashboard needs to calculate flow metrics.
        4. **Exclude unnecessary fields**: It is recommended to exclude the 'Summary' or 'Title' field. These fields are not used and can slow down the upload.
        5. **Protect sensitive data**: Do not include any columns that contain Personally Identifiable Information (PII) or other sensitive data in your export.
        """)
    st.header("🔒 Data Security & Privacy")
    st.success("**Your data is safe.** This application processes your CSV file entirely within your browser. No data from your uploaded file is ever sent to, saved, or stored on any server. When you close this browser tab, your data is permanently discarded.")

def _apply_date_filter(df: pd.DataFrame, date_col_name: str, date_range: str, custom_start_date, custom_end_date) -> pd.DataFrame:
    if date_range == "All time" or pd.to_datetime(df[date_col_name], errors='coerce').isna().all(): return df
    today = pd.to_datetime(datetime.now().date())
    
    start_date = None
    end_date = today

    if date_range == "Last 30 days":
        start_date = today - pd.DateOffset(days=30)
    elif date_range == "Last 60 days":
        start_date = today - pd.DateOffset(days=60)
    elif date_range == "Last 90 days":
        start_date = today - pd.DateOffset(days=90)
    elif date_range == "Custom" and custom_start_date and custom_end_date:
        start_date = pd.to_datetime(custom_start_date)
        end_date = pd.to_datetime(custom_end_date)
    else:
        return df

    # Normalize start to beginning of day and end to end of day for accurate filtering
    start_date = start_date.normalize()
    end_date = end_date.normalize() + pd.Timedelta(days=1, nanoseconds=-1)
    
    return df[(df[date_col_name] >= start_date) & (df[date_col_name] <= end_date)]

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