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

# Configuration
st.set_page_config(page_title="Flow Metrics Dashboard", layout="wide")

class ColorManager:
    """Manages color palettes for the dashboard."""
    DEFAULT_COLORS = {
        'Epic': '#8B5CF6', 'Story': '#10B981', 'Task': '#3B82F6',
        'Bug': '#EF4444', 'Spike': '#F97316'
    }
    # New Color-Blind Friendly palette from ColorBrewer's "PuOr"
    COLOR_BLIND_FRIENDLY_COLORS = {
        'Epic': 'rgb(230,97,1)',
        'Story': 'rgb(253,184,99)',
        'Task': 'rgb(178,171,210)',
        'Bug': 'rgb(94,60,153)',
        'Spike': '#b15928' # Brown from a different palette for Spike
    }
    DEFAULT_PERCENTILE_COLORS = { 50: "red", 70: "orange", 85: "green", 95: "blue" }
    COLOR_BLIND_FRIENDLY_PERCENTILE_COLORS = {
        50: '#E69F00', # Orange
        70: '#d95f02', # Darker Orange
        85: '#7570b3', # Purple
        95: '#e7298a'  # Pink
    }
    DEFAULT_FORECAST_BOX_COLORS = {
        50: "#f8d7da", 70: "#fff3cd", 85: "#d4edda", 95: "#a3bde0"
    }
    COLOR_BLIND_FRIENDLY_FORECAST_BOX_COLORS = {
        50: "#FADADD", # Pastel Pink/Red
        70: "#FFF8DC", # Pastel Yellow/Cornsilk
        85: "#D4E6F1", # Pastel Blue
        95: "#D1E8E2"  # Pastel Teal/Green
    }
    STABILITY_SCORE_COLORS = {
        "Stable": "#28a745", # Green
        "Some Variability": "#ffc107", # Orange/Yellow
        "High Variability": "#dc3545" # Red
    }


    @staticmethod
    def get_work_type_colors(is_color_blind_mode: bool) -> Dict[str, str]:
        """Returns the appropriate color palette for work item types."""
        return ColorManager.COLOR_BLIND_FRIENDLY_COLORS if is_color_blind_mode else ColorManager.DEFAULT_COLORS

    @staticmethod
    def get_percentile_colors(is_color_blind_mode: bool) -> Dict[int, str]:
        """Returns the appropriate color palette for percentile lines."""
        return ColorManager.COLOR_BLIND_FRIENDLY_PERCENTILE_COLORS if is_color_blind_mode else ColorManager.DEFAULT_PERCENTILE_COLORS

    @staticmethod
    def get_forecast_box_colors(is_color_blind_mode: bool) -> Dict[int, str]:
        """Returns the appropriate color palette for forecast likelihood boxes."""
        return ColorManager.COLOR_BLIND_FRIENDLY_FORECAST_BOX_COLORS if is_color_blind_mode else ColorManager.DEFAULT_FORECAST_BOX_COLORS

class Config:
    """Centralized configuration for the dashboard."""
    WORK_TYPE_ORDER = ['Epic', 'Story', 'Task', 'Bug', 'Spike']
    DATE_RANGES = ["All time", "Last 30 days", "Last 60 days", "Last 90 days", "Custom"]

    PERCENTILES = [50, 70, 85, 95]

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
    DEFAULT_COLOR = '#808080'

class ChartConfig:
    """Centralized configuration for chart templates and layouts."""
    CYCLE_TIME_HOVER = (
        "%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br>"
        "<b>Completed:</b> %{customdata[2]}<br><b>Start:</b> %{customdata[3]}<br>"
        "<b>Cycle time:</b> %{customdata[4]} days<extra></extra>"
    )
    BUBBLE_CHART_HOVER = (
        "<b>Items:</b> %{marker.size}<br>"
        "<b>Cycle Time:</b> %{y} days<br>"
        "<b>Completed Date:</b> %{x|%d/%m/%Y}<br><br>"
        "<b>Breakdown:</b><br>%{customdata[0]}<br><br>"
        "<b>Keys:</b><br>%{customdata[1]}<extra></extra>"
    )
    STORY_POINT_HOVER = (
        "<b>Key:</b> %{customdata[0]}<br>"
        "<b>Cycle time:</b> %{y:.0f} days<br>"
        "<b>Story Point:</b> %{x}<br>"
        "<b>Date completed:</b> %{customdata[1]}<extra></extra>"
    )
    AGE_CHART_HOVER = (
        "%{customdata[0]}<br><b>Work type:</b> %{customdata[1]}<br><b>Status:</b> %{customdata[2]}<br>"
        "<b>Start:</b> %{customdata[3]}<br><b>Age:</b> %{customdata[4]} days<extra></extra>"
    )
    WIP_CHART_HOVER = "<b>Date:</b> %{x|%d/%m/%Y}<br><b>WIP count:</b> %{y}<br><b>Breakdown:</b><br>%{customdata[0]}<extra></extra>"
    THROUGHPUT_CHART_HOVER = (
        "<b>Period ending = %{customdata[0]}</b><br>"
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
                return False, "The selected 'Start Status' must come before the 'Done Status' in your workflow. Please check the order of your status columns in the data file."
            return True, ""
        except ValueError:
            return False, "Invalid status columns selected."


class DataProcessor:
    """Handles loading and processing of JIRA export data."""
    @staticmethod
    @st.cache_data
    def load_data(uploaded_file) -> Optional[DataFrame]:
        """Loads data from the uploaded CSV file."""
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, keep_default_na=False, encoding=encoding)
                df = df.dropna(how='all')

                if 'Issue type' in df.columns:
                    df = df.rename(columns={'Issue type': 'Work type'})
                elif 'Issue Type' in df.columns:
                    df = df.rename(columns={'Issue Type': 'Work type'})

                if not {'Key', 'Work type'}.issubset(df.columns):
                    st.error("**Invalid File Format:** The uploaded CSV is missing required columns. Please ensure your file includes both a 'Key' and a 'Work type' column.")
                    return None
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error loading data with encoding {encoding}: {str(e)}")
                return None
        st.error("**File Load Error:** The dashboard could not read this file. Please ensure the file is a standard CSV and try exporting it from Jira again.")
        return None

    @staticmethod
    def clean_data(df: DataFrame) -> DataFrame:
        """Cleans the raw DataFrame, handling duplicates."""
        df_clean = df.copy()

        if df_clean.duplicated(subset=['Key']).any():
            duplicates = df_clean[df_clean.duplicated(subset=['Key'], keep=False)]
            st.warning(
                f"**Data Quality Note:** Found and removed {len(duplicates.drop_duplicates(subset=['Key']))} duplicate work items to prevent calculation errors. The first version of each item was kept. (Example duplicate key: {duplicates['Key'].iloc[0]})"
            )

        df_clean = df_clean.drop_duplicates(subset=['Key'], keep='first').copy()

        def normalize_text(text):
            if not isinstance(text, str):
                return text
            return unicodedata.normalize('NFKC', text).strip()

        for col in ['Key', 'Work type', 'Status']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(normalize_text)

        return df_clean

    @staticmethod
    def process_dates(df: DataFrame, start_col: Optional[str], completed_col: Optional[str]) -> Optional[DataFrame]:
        """Processes date columns and calculates cycle time."""
        try:
            processed_df = df.copy()
            processed_df['Start date'] = processed_df[start_col].apply(DataProcessor._extract_earliest_date) if start_col else pd.NaT
            processed_df['Completed date'] = processed_df[completed_col].apply(DataProcessor._extract_latest_date) if completed_col else pd.NaT
            return DataProcessor._calculate_cycle_time(processed_df)
        except Exception as e:
            st.error(f"Error processing dates: {str(e)}")
            return None

    @staticmethod
    def _parse_date_part(part: str) -> Optional[datetime]:
        """Parses a single date string part, trying multiple formats."""
        part = part.strip()
        formats_to_try = [
            '%d/%m/%Y %H:%M',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y',
            '%Y-%m-%d'
        ]
        for fmt in formats_to_try:
            try:
                return datetime.strptime(part, fmt)
            except ValueError:
                continue
        dt = pd.to_datetime(part, errors='coerce')
        return dt if pd.notna(dt) else None

    @staticmethod
    def _extract_earliest_date(date_str: Union[str, float]) -> Optional[datetime]:
        """Extracts the EARLIEST date from a string that may contain multiple dates."""
        if pd.isna(date_str) or str(date_str).strip() in ['-', '', 'nan']:
            return None
        date_parts = str(date_str).split(',')
        valid_dates = [DataProcessor._parse_date_part(part) for part in date_parts]
        valid_dates = [d for d in valid_dates if d is not None]
        return min(valid_dates) if valid_dates else None

    @staticmethod
    def _extract_latest_date(date_str: Union[str, float]) -> Optional[datetime]:
        """Extracts the LATEST date from a string that may contain multiple dates."""
        if pd.isna(date_str) or str(date_str).strip() in ['-', '', 'nan']:
            return None
        date_parts = str(date_str).split(',')
        valid_dates = [DataProcessor._parse_date_part(part) for part in date_parts]
        valid_dates = [d for d in valid_dates if d is not None]
        return max(valid_dates) if valid_dates else None


    @staticmethod
    def _calculate_cycle_time(df: DataFrame) -> DataFrame:
        """Calculates cycle time for items with start and completed dates."""
        df = df[df['Work type'].notna() & (df['Work type'] != '')].copy()
        df['Cycle time'] = (df['Completed date'] - df['Start date']).dt.days + 1
        invalid_cycle = (df['Cycle time'].notna()) & (df['Cycle time'] < 1)
        if invalid_cycle.any():
            st.warning(f"""
            **Data Quality Warning:** {invalid_cycle.sum()} item(s) were excluded from this chart because their cycle time was calculated as zero or a negative value.
            This usually happens when an item's Start Date is recorded as being after its Completion Date in your data file.
            """)
            df = df[~invalid_cycle]
        return df


class ChartGenerator:
    """Generates Plotly charts for the dashboard."""
    @staticmethod
    @st.cache_data
    def create_cumulative_flow_diagram(df: DataFrame, selected_statuses: List[str], status_col_map: Dict[str, str], date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime], is_color_blind_mode: bool) -> Optional[Figure]:
        """Creates a Cumulative Flow Diagram (CFD)."""
        if not selected_statuses or df.empty:
            return None

        status_cols = [status_col_map[s] for s in selected_statuses]

        id_vars = ['Key']
        cfd_df_melted = df.melt(id_vars=id_vars, value_vars=status_cols, var_name='Status Column', value_name='Date Value')

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
        cfd_cumulative = cfd_pivot.cumsum()

        cfd_cumulative = cfd_cumulative.reindex(full_date_range.date, method='ffill').fillna(0)

        plot_df = cfd_cumulative.reset_index().rename(columns={'index': 'Date'})
        plot_df = plot_df.melt(id_vars='Date', value_name='Count', var_name='Status')

        plot_df['Status'] = pd.Categorical(plot_df['Status'], categories=selected_statuses, ordered=True)
        
        # Use the appropriate color map
        color_map = ColorManager.get_work_type_colors(is_color_blind_mode) if is_color_blind_mode else None

        fig = px.area(plot_df, x='Date', y='Count', color='Status',
                      title='Cumulative Flow Diagram',
                      labels={'Date': 'Date', 'Count': 'Cumulative Count of Items'},
                      category_orders={'Status': selected_statuses},
                      color_discrete_map=color_map)

        fig.update_layout(height=600, legend_title='Workflow Stage')
        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_chart(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        """Creates the cycle time scatterplot with vertical jitter."""
        completed_df = df.dropna(subset=['Start date', 'Completed date', 'Cycle time'])
        if completed_df.empty:
            return None

        chart_df = ChartGenerator._prepare_chart_data(completed_df, ['Key', 'Work type', 'Completed date', 'Start date', 'Cycle time'])

        jitter_strength = 0.4
        chart_df['Cycle_time_jittered'] = chart_df['Cycle time'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(chart_df))

        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scatter(
                x=df_type['Completed date'], y=df_type['Cycle_time_jittered'],
                mode='markers', name=work_type,
                marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7),
                customdata=df_type[['Key', 'Work type', 'Completed_date_formatted', 'Start_date_formatted', 'Cycle_time_formatted']],
                hovertemplate=ChartConfig.CYCLE_TIME_HOVER
            ))

        fig.update_layout(title="Cycle Time Scatterplot", xaxis_title="Completed Date", yaxis_title="Cycle Time (Days)", height=900, legend_title="Work Type")

        ChartGenerator._add_percentile_lines(fig, chart_df, 'Cycle time', chart_df["Completed date"], percentile_settings, is_color_blind_mode)

        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_bubble_chart(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        """Creates an aggregated bubble chart for cycle times."""
        completed_df = df.dropna(subset=['Completed date', 'Cycle time', 'Work type'])
        if completed_df.empty:
            return None

        def format_breakdown(series):
            return '<br>'.join(f"{name}: {count}" for name, count in series.value_counts().items())

        agg_df = completed_df.groupby(['Completed date', 'Cycle time']).agg(
            item_count=('Key', 'size'),
            keys=('Key', lambda x: '<br>'.join(x)),
            breakdown=('Work type', format_breakdown)
        ).reset_index()

        fig = go.Figure(data=[go.Scatter(
            x=agg_df['Completed date'],
            y=agg_df['Cycle time'],
            mode='markers+text',
            text=agg_df['item_count'],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            marker=dict(
                size=agg_df['item_count'],
                sizemode='area',
                sizeref=2.*agg_df['item_count'].max()/(40.**2),
                sizemin=4,
                color='#3B82F6',
                opacity=0.7
            ),
            customdata=agg_df[['breakdown', 'keys']],
            hovertemplate=ChartConfig.BUBBLE_CHART_HOVER
        )])

        fig.update_layout(
            title="Aggregated Cycle Time Bubble Chart",
            xaxis_title="Completed Date",
            yaxis_title="Cycle Time (Days)",
            height=900
        )

        ChartGenerator._add_percentile_lines(fig, completed_df, 'Cycle time', agg_df["Completed date"], percentile_settings, is_color_blind_mode)

        return fig

    @staticmethod
    @st.cache_data
    def create_cycle_time_box_plot(df: DataFrame, interval: str, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
        """Creates a box plot of cycle times over a given interval."""
        df_completed = df.dropna(subset=['Completed date', 'Cycle time']).copy()
        if df_completed.empty:
            return None

        freq_map = {'Weekly': 'W-MON', 'Monthly': 'M'}
        df_completed['Period'] = df_completed['Completed date'].dt.to_period(freq_map[interval]).dt.start_time

        fig = px.box(
            df_completed,
            x='Period',
            y='Cycle time',
            title=f'Cycle Time Distribution per {interval}',
            labels={'Period': interval, 'Cycle time': 'Cycle Time (Days)'},
            points='all'
        )
        
        # Apply the new date formatting
        if interval == 'Monthly':
            fig.update_xaxes(tickformat="%b %Y")
        else:
            fig.update_xaxes(tickformat="%d %b %Y")
            
        fig.update_layout(height=900)

        ChartGenerator._add_percentile_lines(fig, df_completed, 'Cycle time', df_completed["Period"], percentile_settings, is_color_blind_mode, add_annotation=True)

        return fig


    @staticmethod
    @st.cache_data
    def create_cycle_time_histogram(df: DataFrame, percentile_settings: Dict[str, bool], is_color_blind_mode: bool) -> Optional[Figure]:
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
        fig.update_layout(bargap=0.1, yaxis_title="Number of Work Items", height=600)

        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for p, color in percentile_colors.items():
            if percentile_settings.get(f"show_{p}th", True):
                percentile_val = np.percentile(completed_df['Cycle time'], p)
                fig.add_vline(
                    x=percentile_val, line_dash="dash", line_color=color,
                    annotation_text=f"{p}th: {int(percentile_val)}d",
                    annotation_position="top right"
                )
        return fig

    @staticmethod
    @st.cache_data
    def create_time_in_status_chart(df: DataFrame, status_cols: List[str]) -> Tuple[Optional[Figure], Optional[DataFrame]]:
        """Calculates and creates a bar chart of the average time spent in each status."""

        if len(status_cols) < 2 or df.empty:
            return None, None

        all_durations = []
        for i in range(len(status_cols) - 1):
            current_col = status_cols[i]
            next_col = status_cols[i+1]

            temp_df = df[[current_col, next_col]].copy()

            temp_df['current_date'] = temp_df[current_col].apply(DataProcessor._extract_latest_date)
            temp_df['next_date'] = temp_df[next_col].apply(DataProcessor._extract_latest_date)

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
            height=600,
            yaxis_range=[0, chart_df['Average Time (Days)'].max() * 1.15]
        )
        return fig, chart_df

    @staticmethod
    def create_work_item_age_chart(
        plot_df: DataFrame,
        wip_df: DataFrame,
        status_order: List[str],
        cycle_time_percentiles: Dict[str, int],
        percentile_settings: Dict[str, bool],
        is_color_blind_mode: bool
    ) -> Optional[Figure]:
        """
        Creates the work item age chart with a column-based layout and horizontal jitter.
        Uses a separate DataFrame for WIP counts to ensure accuracy.
        """
        age_data = []
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

        # chart_df contains only the items to be PLOTTED as dots
        chart_df = pd.DataFrame(age_data) if age_data else pd.DataFrame(columns=['Key', 'Work type', 'Status', 'Age', 'Start date'])

        if not chart_df.empty:
            status_map = {status: i for i, status in enumerate(status_order)}
            chart_df['Status_Num'] = chart_df['Status'].map(status_map)
            jitter_strength = 0.25
            chart_df['Status_Jittered'] = chart_df['Status_Num'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(chart_df))
            chart_df = ChartGenerator._prepare_chart_data(chart_df, ['Key', 'Work type', 'Status', 'Age', 'Status_Jittered', 'Start date'])
            chart_df.dropna(subset=['Status', 'Status_Jittered'], inplace=True)

        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()

        if not chart_df.empty:
            for work_type in ChartGenerator._order_work_types(chart_df):
                df_type = chart_df[chart_df['Work type'] == work_type]
                fig.add_trace(go.Scatter(
                    x=df_type['Status_Jittered'],
                    y=df_type['Age'],
                    mode='markers',
                    name=work_type,
                    marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7),
                    customdata=df_type[['Key', 'Work type', 'Status', 'Start_date_formatted', 'Age_formatted']],
                    hovertemplate=ChartConfig.AGE_CHART_HOVER
                ))

        max_age_plot = chart_df['Age'].max() if not chart_df.empty else 10
        y_axis_max = max_age_plot * 1.15

        fig.update_layout(
            title="Work Item Age Analysis",
            yaxis_title="<b>Age (Calendar Days)</b>",
            height=900, legend_title="Work Type",
            xaxis=dict(
                title_text="",
                tickvals=list(range(len(status_order))),
                ticktext=[f"<b>{s}</b>" for s in status_order],
                tickfont=dict(size=14),
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey',
                tickfont=dict(size=14),
                title_font=dict(size=16),
                range=[0, y_axis_max]
            ),
            showlegend=True
        )

        for i, status in enumerate(status_order):
            if i > 0:
                fig.add_vline(x=i - 0.5, line_width=2, line_color='LightGrey')

            count = len(wip_df[wip_df['Status'] == status])
            fig.add_annotation(
                x=i, y=y_axis_max, text=f"<b>WIP = {count}</b>",
                showarrow=False, font=dict(size=14, color="black"),
                yanchor="bottom", yshift=5
            )
        
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
        """Creates a scatter plot of cycle time vs. story points with specific axis formatting."""
        sp_col_name = None
        if 'Story Points' in df.columns:
            sp_col_name = 'Story Points'
        elif 'Story point estimate' in df.columns:
            sp_col_name = 'Story point estimate'

        if not sp_col_name:
            return None

        df_sp = df.dropna(subset=['Cycle time', sp_col_name]).copy()
        df_sp = df_sp[pd.to_numeric(df_sp[sp_col_name], errors='coerce').notna()]
        df_sp[sp_col_name] = pd.to_numeric(df_sp[sp_col_name])

        if df_sp.empty:
            return None

        chart_df = ChartGenerator._prepare_chart_data(df_sp, ['Key', 'Work type', 'Completed date', 'Start date', 'Cycle time'])
        chart_df[sp_col_name] = df_sp[sp_col_name]

        jitter_strength = 0.4
        chart_df['Cycle_time_jittered'] = chart_df['Cycle time'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(chart_df))

        work_type_colors = ColorManager.get_work_type_colors(is_color_blind_mode)
        fig = go.Figure()
        for work_type in ChartGenerator._order_work_types(chart_df):
            df_type = chart_df[chart_df['Work type'] == work_type]
            fig.add_trace(go.Scatter(
                x=df_type[sp_col_name], y=df_type['Cycle_time_jittered'],
                mode='markers', name=work_type,
                marker=dict(color=work_type_colors.get(work_type, Config.DEFAULT_COLOR), size=8, opacity=0.7),
                customdata=df_type[['Key', 'Completed_date_formatted']],
                hovertemplate=ChartConfig.STORY_POINT_HOVER
            ))

        all_ticks = [1, 2, 3, 5, 8, 13, 20, 40, 100]
        max_sp_value = chart_df[sp_col_name].max()
        visible_ticks = [t for t in all_ticks if t <= max_sp_value]
        if not visible_ticks or max_sp_value > visible_ticks[-1]:
             if not any(abs(max_sp_value - t) < 0.1 for t in visible_ticks):
                visible_ticks.append(int(np.ceil(max_sp_value)))
                visible_ticks.sort()

        tick_labels = [f"<b>{t}</b>" for t in visible_ticks]

        fig.update_layout(
            title="Story Point Correlation",
            xaxis_title="Story Points",
            yaxis_title="Cycle Time (Days)",
            height=900,
            legend_title="Work Type",
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.02)
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=visible_ticks,
            ticktext=tick_labels,
            tickfont=dict(size=14)
        )

        return fig

    @staticmethod
    def _add_percentile_lines(fig: go.Figure, df: pd.DataFrame, y_col: str, x_data, percentile_settings: Dict[str, bool], is_color_blind_mode: bool, add_annotation: bool = False):
        """Helper to add percentile lines to a chart."""
        if df.empty:
            return

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
    @st.cache_data
    def create_wip_chart(df: DataFrame, date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime]) -> Optional[Figure]:
        """Creates the WIP (Work In Progress) run chart."""
        if df is None or df.empty: return None
        wip_df = df.dropna(subset=['Start date'])
        if wip_df.empty: return None

        plot_min_date = wip_df['Start date'].min()
        latest_start = wip_df['Start date'].max()
        latest_completion = wip_df['Completed date'].max()
        
        # Determine the end date for the chart's x-axis
        plot_max_date = max(dt for dt in [latest_start, latest_completion] if pd.notna(dt))

        # Ensure the date range doesn't go into the future unnecessarily
        today = pd.to_datetime(datetime.now().date())
        if plot_max_date > today:
            plot_max_date = today

        all_dates = pd.date_range(start=plot_min_date, end=plot_max_date, freq='D')
        
        filtered_dates_df = _apply_date_filter(pd.DataFrame({'Date': all_dates}), 'Date', date_range, custom_start_date, custom_end_date)
        if filtered_dates_df.empty: return None
        
        filtered_dates = filtered_dates_df['Date']
        
        daily_wip_data = []
        for day in filtered_dates:
            daily_wip_df = wip_df[
                (wip_df['Start date'] <= day) &
                ((wip_df['Completed date'].isna()) | (wip_df['Completed date'] > day))
            ]
            
            breakdown_str = '<br>'.join(f"{wt}: {count}" for wt, count in daily_wip_df['Work type'].value_counts().items())
            daily_wip_data.append({'Date': day, 'WIP': len(daily_wip_df), 'Breakdown': breakdown_str})
        
        if not daily_wip_data: return None
        wip_over_time = pd.DataFrame(daily_wip_data)
        
        fig = px.line(wip_over_time, x="Date", y="WIP", title="WIP (Work In Progress) Run Chart")
        fig.update_traces(
            customdata=wip_over_time[['Breakdown']],
            hovertemplate=ChartConfig.WIP_CHART_HOVER
        )
        
        fig.update_layout(height=600)
        ChartGenerator._add_trend_line(fig, wip_over_time)
        return fig

    @staticmethod
    @st.cache_data
    def create_throughput_chart(df: DataFrame, interval: str, throughput_status_col: str, date_range: str, custom_start_date: Optional[datetime], custom_end_date: Optional[datetime], sprint_anchor_date: Optional[datetime.date] = None, overall_max_date: Optional[datetime] = None) -> Optional[Tuple[Figure, DataFrame]]:
        """Creates the throughput bar chart."""
        if not throughput_status_col:
            return None, None

        throughput_df = df.copy()
        throughput_df['Throughput Date'] = throughput_df[throughput_status_col].apply(DataProcessor._extract_latest_date)
        throughput_df.dropna(subset=['Throughput Date'], inplace=True)

        if throughput_df.empty:
            return None, None
        
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
            
            throughput_df['Period Interval'] = pd.cut(throughput_df['Throughput Date'], bins=bins, right=True, include_lowest=True, labels=bins[1:])
            
            throughput_df.dropna(subset=['Period Interval'], inplace=True)

            agg_df = throughput_df.groupby('Period Interval').agg(
                Throughput=('Key', 'count'),
                Details=('Work type', lambda s: '<br>'.join(f"{wt}: {count}" for wt, count in s.value_counts().items()))
            ).reset_index()

            agg_df['Period End'] = pd.to_datetime(agg_df['Period Interval'])
            agg_df['Period Start'] = agg_df['Period End'] - pd.DateOffset(days=13)

        else: # Weekly or Monthly logic
            freq_string = 'W-MON' if interval == 'Weekly' else 'MS'
            grouper = pd.Grouper(key='Throughput Date', freq=freq_string)
            agg_df = throughput_df.groupby(grouper).agg(
                Throughput=('Key', 'count'),
                Details=('Work type', lambda s: '<br>'.join(f"{wt}: {count}" for wt, count in s.value_counts().items()))
            ).reset_index()
            
            agg_df.rename(columns={'Throughput Date': 'Period Start'}, inplace=True)
            
            if interval == 'Weekly':
                agg_df['Period End'] = agg_df['Period Start'] + pd.DateOffset(days=6)
            else: # Monthly
                agg_df['Period End'] = agg_df['Period Start'] + pd.offsets.MonthEnd(0)

        agg_df = _apply_date_filter(agg_df, 'Period End', date_range, custom_start_date, custom_end_date)
        if agg_df.empty: return None, None
        
        if overall_max_date:
            agg_df = agg_df[agg_df['Period Start'] <= overall_max_date]
        if agg_df.empty: return None, None
        
        agg_df = agg_df.sort_values(by='Period Start')
        
        agg_df['Period_End_Formatted'] = agg_df['Period End'].dt.strftime('%d/%m/%Y')
        agg_df['Details'] = "<b>Breakdown:</b><br>" + agg_df['Details']

        title_interval = interval.replace("ly", "")
        
        if interval == 'Monthly':
            agg_df['Period Label'] = agg_df['Period End'].dt.strftime('%b %Y')
        else:
            agg_df['Period Label'] = agg_df['Period End'].dt.strftime('%d %b %Y')

        fig = px.bar(agg_df, x='Period Label', y="Throughput", title=f"Throughput per {title_interval}", text="Throughput")
        fig.update_traces(
            textposition='outside',
            hovertemplate=ChartConfig.THROUGHPUT_CHART_HOVER,
            customdata=agg_df[['Period_End_Formatted', 'Details']].values
        )

        fig.update_layout(
            height=600, 
            yaxis_range=[0, agg_df['Throughput'].max() * 1.15], 
            xaxis_title="Period Ending",
        )
        
        fig.update_xaxes(categoryorder="array", categoryarray=agg_df['Period Label'].tolist())

        return fig, agg_df

    @staticmethod
    @st.cache_data(show_spinner="Running simulations...")
    def _get_recent_weekly_throughput(df: DataFrame, status_col: str) -> Tuple[Optional[pd.Series], Optional[np.ndarray]]:
        """Gets recent weekly throughput and calculates sampling weights."""
        if not status_col:
            return None, None
        forecast_df = df.copy()
        forecast_df['Forecast Completion Date'] = forecast_df[status_col].apply(DataProcessor._extract_latest_date)
        completed_df = forecast_df.dropna(subset=['Forecast Completion Date'])

        if len(completed_df) < 2:
            return None, None
        
        last_completion_date = completed_df['Forecast Completion Date'].max()
        start_of_period = last_completion_date - pd.DateOffset(weeks=25)
        recent_completed_df = completed_df[completed_df['Forecast Completion Date'] > start_of_period]
        if recent_completed_df.empty:
            st.warning("**Forecasting requires more data.** To run a forecast, the dashboard needs at least two weeks of completed work from the last 25 weeks.")
            return None, None

        weekly_throughput = recent_completed_df.groupby(pd.Grouper(key='Forecast Completion Date', freq='W-MON')).size()
        
        max_date = recent_completed_df['Forecast Completion Date'].max()
        weekly_throughput = weekly_throughput[weekly_throughput.index <= max_date]

        num_weeks_of_data = len(weekly_throughput)
        if num_weeks_of_data < 2:
            st.warning("**Forecasting requires more data.** To run a forecast, the dashboard needs at least two weeks of completed work from the last 25 weeks.")
            return None, None
        if num_weeks_of_data < 7:
            st.warning(f"**Note:** This forecast is based on only {num_weeks_of_data} weeks of data. Forecasts become more reliable with more history.")

        weights = np.arange(1, num_weeks_of_data + 1)
        normalized_weights = weights / np.sum(weights)

        return weekly_throughput, normalized_weights


    @staticmethod
    @st.cache_data(show_spinner="Running 'How Many' simulations...")
    def create_how_many_forecast_chart(df: DataFrame, forecast_days: int, throughput_status_col: str, is_color_blind_mode: bool) -> Optional[Figure]:
        """Prepares data and runs the 'How Many' simulation to create a forecast chart."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None:
            return None

        num_weeks = forecast_days / 7.0
        num_full_weeks = int(num_weeks)
        fractional_week_multiplier = num_weeks % 1

        simulations = np.random.choice(
            weekly_throughput, size=(Config.FORECASTING_SIMULATIONS, num_full_weeks),
            replace=True, p=normalized_weights
        )

        forecast_counts = simulations.sum(axis=1)

        if fractional_week_multiplier > 0:
            last_week_sim = np.random.choice(weekly_throughput, size=Config.FORECASTING_SIMULATIONS, replace=True, p=normalized_weights)
            forecast_counts += (last_week_sim * fractional_week_multiplier).astype(int)

        counts, bin_edges = np.histogram(forecast_counts, bins=30, range=(forecast_counts.min(), forecast_counts.max()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = go.Figure(data=[go.Bar(x=bin_centers, y=counts, name='Simulations')])
        fig.update_layout(
            title=f"Forecast: How Many Items in the Next {forecast_days} Days?",
            xaxis_title="Number of Items Completed", yaxis_title="Frequency",
            bargap=0.1, yaxis_range=[0, counts.max() * 1.20], height=600
        )

        summary_text = f"**Forecast Summary (for next {forecast_days} days):**"
        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for likelihood, percentile in sorted(Config.FORECAST_LIKELIHOODS.items(), reverse=True):
            value = np.percentile(forecast_counts, percentile)
            color_key = next((k for k, v in Config.FORECAST_LIKELIHOODS.items() if v == percentile), 50)
            color = percentile_colors.get(color_key)
            fig.add_vline(
                x=value, line_dash="dash", line_color=color,
                annotation_text=f"{likelihood}%: {int(value)}",
                annotation_position="top left"
            )
            summary_text += f"\n- There is a **{likelihood}% chance** to complete **{int(value)} or more** items."

        st.markdown(summary_text)

        return fig

    @staticmethod
    @st.cache_data(show_spinner="Running 'When' simulations...")
    def create_when_forecast_chart(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str, scope_complexity: str, team_focus: str, is_color_blind_mode: bool) -> Tuple[Optional[Figure], Optional[Dict[int, datetime]]]:
        """Creates the 'When' forecast chart by running a direct simulation."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None or normalized_weights is None or weekly_throughput.mean() == 0:
            return None, None

        complexity_factors = {
            'Clear and understood': 1.0, 'Somewhat understood': 1.25,
            'Not really understood yet': 1.50, 'Very unclear or not understood': 2.00
        }
        adjusted_items = int(items_to_complete * complexity_factors.get(scope_complexity, 1.0))

        focus_factors = {
            '100% (only this work)': 1.0, '75% (mostly this work)': 0.75,
            '50% (half of this work)': 0.50, '25% (some of this work)': 0.25
        }
        adjusted_throughput = weekly_throughput * focus_factors.get(team_focus, 1.0)

        completion_weeks_data = []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            items_done = 0
            weeks_elapsed = 0
            timeout_weeks = max(300, (adjusted_items / adjusted_throughput.mean()) * 20 if adjusted_throughput.mean() > 0 else 300)
            while items_done < adjusted_items:
                if weeks_elapsed > timeout_weeks:
                    weeks_elapsed = -1
                    break
                items_done += np.random.choice(adjusted_throughput, p=normalized_weights)
                weeks_elapsed += 1
            if weeks_elapsed != -1:
                completion_weeks_data.append(weeks_elapsed)

        if not completion_weeks_data:
            return None, None

        completion_days_data = [w * 7 for w in completion_weeks_data]

        value_counts = pd.Series(completion_days_data).value_counts().sort_index()
        fig = go.Figure(data=[go.Bar(name='Simulations', x=value_counts.index, y=value_counts.values)])
        fig.update_layout(
            title="Forecast: Completion Date Distribution",
            xaxis_title=f"Days from {start_date.strftime('%d %b, %Y')} to Completion",
            yaxis_title="Frequency (Number of Simulations)",
            bargap=0.5, height=600
        )

        percentile_dates = {}
        percentile_colors = ColorManager.get_percentile_colors(is_color_blind_mode)
        for p in Config.PERCENTILES:
            days = np.percentile(completion_days_data, p)
            percentile_dates[p] = start_date + timedelta(days=int(days))
            fig.add_vline(
                x=days, line_dash="dash", line_color=percentile_colors.get(p),
                annotation_text=f"{p}%", annotation_position="top right"
            )

        return fig, percentile_dates

    @staticmethod
    @st.cache_data
    def run_when_scenario_forecast(df: DataFrame, items_to_complete: int, start_date: datetime.date, throughput_status_col: str) -> Optional[Dict]:
        """Runs the good week/bad week scenario analysis and returns the results."""
        weekly_throughput, normalized_weights = ChartGenerator._get_recent_weekly_throughput(df, throughput_status_col)
        if weekly_throughput is None or normalized_weights is None or weekly_throughput.mean() == 0:
            return None

        median_throughput = weekly_throughput.median()
        good_weeks = weekly_throughput[weekly_throughput > median_throughput]
        bad_weeks = weekly_throughput[weekly_throughput <= median_throughput]

        if good_weeks.empty or bad_weeks.empty:
            return None

        good_completion_weeks = []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            items_done = np.random.choice(good_weeks)
            weeks_elapsed = 1
            while items_done < items_to_complete:
                items_done += np.random.choice(weekly_throughput, p=normalized_weights)
                weeks_elapsed += 1
            good_completion_weeks.append(weeks_elapsed * 7)

        bad_completion_weeks = []
        for _ in range(Config.FORECASTING_SIMULATIONS):
            items_done = np.random.choice(bad_weeks)
            weeks_elapsed = 1
            while items_done < items_to_complete:
                items_done += np.random.choice(weekly_throughput, p=normalized_weights)
                weeks_elapsed += 1
            bad_completion_weeks.append(weeks_elapsed * 7)

        results = {}
        for p in Config.PERCENTILES:
            good_day = np.percentile(good_completion_weeks, p)
            bad_day = np.percentile(bad_completion_weeks, p)
            results[p] = {
                'Good Week Start': (start_date + timedelta(days=int(good_day))).strftime('%d %b, %Y'),
                'Bad Week Start': (start_date + timedelta(days=int(bad_day))).strftime('%d %b, %Y')
            }
        return results

    @staticmethod
    def _prepare_chart_data(df: DataFrame, columns: List[str]) -> DataFrame:
        """Prepares a DataFrame for charting by formatting columns."""
        chart_df = df.copy()
        for col in ['Completed date', 'Start date']:
            if col in chart_df.columns and not chart_df[col].dropna().empty:
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
        st.cache_data.clear()
        st.markdown("""
            <style>
                [data-testid="stTabs"] button {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px 15px;
                }
            </style>
        """, unsafe_allow_html=True)
        with st.spinner("🔄 Processing JIRA export..."):
            loaded_df = DataProcessor.load_data(self.uploaded_file)
            if loaded_df is None: return
            self.raw_df = DataProcessor.clean_data(loaded_df)

        self.status_mapping = StatusManager.extract_status_columns(self.raw_df)
        if not self.status_mapping:
            st.error("**Configuration Error:** Could not find any status columns in your file. Please make sure your Jira export was created with the 'Show entry dates' option selected, which creates the required '-> Status' columns.")
            return

        date_bounds_df = self._pre_process_for_sidebar()

        self._display_sidebar(date_bounds_df)

        self._display_charts()

    def _pre_process_for_sidebar(self) -> DataFrame:
        """A light processing step to get all possible dates for sidebar validation."""
        status_cols = list(self.status_mapping.values())
        if not status_cols:
            return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})

        df = self.raw_df.copy()

        all_dates = []
        for col in status_cols:
            dates_in_col = df[col].apply(DataProcessor._extract_latest_date).dropna()
            all_dates.extend(pd.to_datetime(dates_in_col, errors='coerce').dropna())

        if not all_dates:
             return pd.DataFrame({'Start date': [pd.NaT], 'Completed date': [pd.NaT]})

        date_df = pd.DataFrame(all_dates, columns=['Date'])
        return pd.DataFrame({
            'Start date': [date_df['Date'].min()],
            'Completed date': [date_df['Date'].max()]
        })
        
    def _handle_multiselect(self, key):
        """Manages the 'All' option for a multiselect widget in session state."""
        prev_key = f"prev_{key}"
        
        if prev_key not in st.session_state:
            st.session_state[prev_key] = st.session_state[key]
            
        current_selection = st.session_state[key]
        previous_selection = st.session_state[prev_key]

        if current_selection == previous_selection:
            return

        all_was_selected = "All" in previous_selection
        all_is_selected = "All" in current_selection

        if not all_was_selected and all_is_selected:
            st.session_state[key] = ["All"]
        elif all_was_selected and len(current_selection) > 1:
            st.session_state[key] = [s for s in current_selection if s != "All"]
        elif not current_selection:
            st.session_state[key] = ["All"]

        st.session_state[prev_key] = st.session_state[key]


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
        st.sidebar.caption("ℹ️ *In the multi-select filters, choosing 'All' will deselect any other options. Likewise, choosing a specific option will deselect 'All'.*")
        
        # --- Work Item Type Filter with interactive 'All' ---
        work_type_key = 'work_types'
        if work_type_key not in st.session_state:
            st.session_state[work_type_key] = ['All']

        st.sidebar.multiselect(
            "Work Item Type",
            options=["All"] + ChartGenerator._order_work_types(self.raw_df),
            key=work_type_key,
            on_change=self._handle_multiselect,
            args=(work_type_key,),
            help="Select one or more work types."
        )
        self.selections[work_type_key] = st.session_state[work_type_key]
        
        # --- Date Range Filter ---
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

        self.selections["exclude_long_cycle_times"] = st.sidebar.checkbox("Exclude cycle time > 365 days", value=False)
        st.sidebar.caption("Note: Date Range does not apply to the Work Item Age chart.")

        st.sidebar.markdown("#### Optional Filters")
        for f_name, f_type in Config.OPTIONAL_FILTERS.items():
            if f_name in self.raw_df.columns:
                unique_vals = self._get_unique_values(self.raw_df[f_name], f_type)
                if f_type == "single":
                    self.selections[f_name] = st.sidebar.selectbox(f_name, ["All"] + unique_vals, key=f"filter_{f_name}")
                else:
                    session_key = f"selection_{f_name}"
                    if session_key not in st.session_state:
                        st.session_state[session_key] = ['All']
                    
                    st.sidebar.multiselect(
                        f_name, 
                        ["All"] + unique_vals, 
                        key=session_key, 
                        on_change=self._handle_multiselect, 
                        args=(session_key,)
                    )
                    self.selections[f_name] = st.session_state[session_key]

    def _sidebar_chart_controls(self):
        """Controls for customizing individual charts."""
        st.sidebar.markdown("### Accessibility")
        self.selections['color_blind_mode'] = st.sidebar.checkbox("Enable Color-Blind Friendly Mode")

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

        if self.selections.get("exclude_long_cycle_times"):
            if 'Cycle time' in df.columns:
                df = df[df['Cycle time'] <= 365]

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

        if apply_date_filter and 'Completed date' in df.columns:
            df = _apply_date_filter(df, 'Completed date', self.selections["date_range"], self.selections["custom_start_date"], self.selections["custom_end_date"])

        return df

    def _display_header_and_metrics(self, stats: Dict):
        """Displays metrics and filters that depend on a configured cycle time."""
        st.info(f"📊 **Cycle Time Configuration:** Starting: **{self.selections['start_status']}** | Done: **{self.selections['completed_status']}**")
        with st.expander("Cycle Time Data Summary"):
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Total Items in Filter", stats['total'])
            c2.metric("✅ Completed Items", stats['completed'])
            c3.metric("🔄 Still In Progress", stats['in_progress'])
        
        st.info("ℹ️ **Cycle Time Formula:** (Done Date - Starting Date) + 1 days")
        st.caption("We add one day to be inclusive. This ensures that an item started and completed on the same day has a cycle time of 1 day, not 0.")

        active_filters = [format_multiselect_display(self.selections['work_types'], 'Work types')]
        for f_name, f_type in Config.OPTIONAL_FILTERS.items():
            selection = self.selections.get(f_name)
            if isinstance(selection, list) and "All" not in selection and selection:
                active_filters.append(f"{f_name}: {format_multiselect_display(selection, '')}")
            elif isinstance(selection, str) and selection != "All":
                 active_filters.append(f"{f_name}: {selection}")

        date_range_display = f"from {self.selections['custom_start_date'].strftime('%Y-%m-%d')} to {self.selections['custom_end_date'].strftime('%Y-%m-%d')}" if self.selections['date_range'] == "Custom" and self.selections['custom_start_date'] and self.selections['custom_end_date'] else self.selections['date_range']
        active_filters.append(date_range_display)
        st.markdown(f"**🎯 Showing:** {' | '.join(filter(None, active_filters))}")

    def _display_charts(self):
        """Displays the main chart area with tabs."""
        tab_list = ["📈 Cycle Time", "🔄 Process Flow", "📊 Work Item Age", "🔄 WIP Trend", "📊 Throughput", "🔮 Throughput Forecast"]

        sp_col_name = None
        if 'Story Points' in self.raw_df.columns:
            sp_col_name = 'Story Points'
        elif 'Story point estimate' in self.raw_df.columns:
            sp_col_name = 'Story point estimate'

        if sp_col_name and pd.to_numeric(self.raw_df[sp_col_name], errors='coerce').notna().any():
            tab_list.append("📈 Story Point Analysis")

        main_tabs = st.tabs(tab_list)

        with main_tabs[0]: self._display_cycle_time_charts()
        with main_tabs[1]: self._display_cfd_chart()
        with main_tabs[2]: self._display_work_item_age_chart()
        with main_tabs[3]: self._display_wip_chart()
        with main_tabs[4]: self._display_throughput_chart()
        with main_tabs[5]: self._display_forecast_charts()
        if len(main_tabs) > 6:
            with main_tabs[6]: self._display_story_point_chart()

    def _display_cfd_chart(self):
        """Displays the Cumulative Flow Diagram and its controls."""
        st.header("Process Stability & Flow")
        st.markdown('<div class="styled-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
             st.markdown("""
                - **What it is:** This chart shows the cumulative number of work items in each stage of your workflow over time.
                - **How to read it:** Each colored area represents a stage in your workflow. The vertical distance between the lines shows the number of items in that stage on a given day.
                - **What patterns to look for:**
                    - **Widening Bands:** If a colored band is getting wider over time, it indicates that more work is arriving in that stage than is leaving it. This is a classic sign of a bottleneck.
                    - **Flat Bands:** If all bands are flat, it means no work is being completed.
                    - **Parallel Bands:** If the top and bottom lines of the chart are moving in parallel, it generally indicates a stable flow.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Cumulative Flow Diagram (CFD) Controls", expanded=True):

            all_statuses = list(self.status_mapping.keys())

            self.selections['cfd_statuses'] = st.multiselect(
                "Select workflow statuses in order",
                options=all_statuses,
                default=all_statuses,
                help="Select the statuses you want to appear in the chart. The order of selection determines the stacking order."
            )

        st.divider()

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
        st.markdown('<div class="styled-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about these charts", icon="🎓"):
            st.markdown("""
                - **What it is:** These charts help visualize the consistency of your team's delivery over time.
                - **How to read it:** Each dot is a completed work item. The vertical position of a dot shows its Cycle Time, and the horizontal position shows its completion date. Percentile lines show the percentage of work items that were completed in that time or less. For example, the 85th percentile line shows the point at which 85% of items were completed.
                - **What patterns to look for:**
                    - **Clusters of dots** can indicate a change in your process or team that affected delivery speed.
                    - **Gaps in the data** (where no dots appear) may suggest that work is being delivered in large batches rather than a smooth flow, often at the end of a release cycle.
                    - **Outliers** (dots with very high Cycle Times) often represent items that were blocked by external dependencies or were too large to begin with.
                """)
        st.markdown('</div>', unsafe_allow_html=True)

        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2, col3 = st.columns(3)

        with col1:
            self.selections["start_status"] = st.selectbox("Starting Status", status_options, key="cycle_time_start")
        with col2:
            self.selections["completed_status"] = st.selectbox("Done Status", status_options, key="cycle_time_end")

        with col3:
            self.selections["box_plot_interval"] = st.selectbox("Box Plot Grouping", ["Weekly", "Monthly"], index=0)

        self.selections["start_col"] = self.status_mapping.get(self.selections["start_status"])
        self.selections["completed_col"] = self.status_mapping.get(self.selections["completed_status"])

        st.divider()

        if not self.selections["start_col"] or not self.selections["completed_col"]:
            st.info("To see your charts, please choose a 'Starting Status' and a 'Done Status' from the dropdowns above.")
            self.filtered_df = pd.DataFrame()
            return

        is_valid, error_msg = StatusManager.validate_status_order(self.raw_df, self.selections["start_col"], self.selections["completed_col"])
        if not is_valid:
            st.error(error_msg)
            return

        self.processed_df = DataProcessor.process_dates(self.raw_df, self.selections["start_col"], self.selections["completed_col"])
        if self.processed_df is None:
             st.warning("Could not calculate Cycle Time with the selected statuses. Please check your selections.")
             return

        self.filtered_df = self._apply_all_filters(self.processed_df, apply_date_filter=True)
        cycle_stats = StatsCalculator.cycle_time_stats(self.filtered_df)
        summary_stats = StatsCalculator.summary_stats(self.filtered_df)
        
        if cycle_stats is None:
            st.warning("No completed items found for the selected criteria. Unable to display Cycle Time charts.")
            self.filtered_df = pd.DataFrame() # Ensure filtered_df is empty so other charts don't use stale data
            return

        self._display_header_and_metrics(summary_stats)

        ct_tabs = st.tabs(["Scatter Plot", "Bubble Chart", "Box Plot", "Distribution (Histogram)", "Time in Status"])
        with ct_tabs[0]:
            st.subheader("Scatter Plot")
            st.markdown("ℹ️ *A small amount of random vertical 'jitter' has been added to separate overlapping points.*")
            chart = ChartGenerator.create_cycle_time_chart(self.filtered_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[1]:
            st.subheader("Aggregated Bubble Chart")
            st.markdown("ℹ️ *Bubbles represent one or more items completed on the same day with the same cycle time.*")
            chart = ChartGenerator.create_cycle_time_bubble_chart(self.filtered_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[2]:
            st.subheader("Cycle Time Distribution Over Time")
            chart = ChartGenerator.create_cycle_time_box_plot(self.filtered_df, self.selections["box_plot_interval"], self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[3]:
            st.subheader("Distribution (Histogram)")
            chart = ChartGenerator.create_cycle_time_histogram(self.filtered_df, self.selections["percentiles"], self.selections['color_blind_mode'])
            if chart: st.plotly_chart(chart, use_container_width=True)
            else: st.warning("No completed items in the selected date range could be found to display on this chart.")
        with ct_tabs[4]:
            st.subheader("Time in Status")
            st.markdown("This chart shows the average time items spend in each status column of your raw data export.")
            status_cols = list(self.status_mapping.values())
            chart, chart_data = ChartGenerator.create_time_in_status_chart(self.filtered_df, status_cols)
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
                st.warning("Not enough data was found for the selected statuses to calculate the average time spent in each.")

    def _display_story_point_chart(self):
        """Displays the Story Point Correlation chart and its controls."""
        st.header("Story Point Analysis")

        st.markdown("This chart plots the cycle time of completed items against their story point estimates.")

        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)

        with col1:
            self.selections["sp_start_status"] = st.selectbox("Starting Status", status_options, key="sp_start")
        with col2:
            self.selections["sp_end_status"] = st.selectbox("Done Status", status_options, key="sp_end", help="Please select the status that represents your Definition of Done.")

        self.selections["sp_start_col"] = self.status_mapping.get(self.selections["sp_start_status"])
        self.selections["sp_end_col"] = self.status_mapping.get(self.selections["sp_end_status"])

        st.divider()

        if not self.selections["sp_start_col"] or not self.selections["sp_end_col"]:
            st.info("ℹ️ Please select a 'Starting Status' and 'Done Status' above to generate the chart.")
            return

        sp_processed_df = DataProcessor.process_dates(self.raw_df, self.selections["sp_start_col"], self.selections["sp_end_col"])
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
        st.markdown('<div class="styled-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
            - **What it is:** This chart shows all the items that are currently in progress, their current status (column), and how long they have been in progress (their Age).
            - **How to read it:** Each dot is a work item that has started but not yet finished. Its vertical position shows its current age in days.
            - **What to look for:**
                - **The oldest items first.** The most important question in a Daily Stand-up is "what's the oldest thing we are working on, and what are we doing to get it moving?"
                - **Items nearing or crossing percentile lines.** The percentile lines are taken from your historical Cycle Time data. If an item's age is approaching the 85th percentile, it is at high risk of taking longer than 85% of all your previous items. This is a crucial signal to the team to intervene by swarming on the item or breaking it down.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("""
        - WIP counts at the top show all in-progress items currently in that status.
        - Dots on the chart represent only those items that have passed through the selected 'Start Status for Age Calculation'.
        - The percentile lines are based on the cycle time of completed items (from the "Cycle Time" tab) to help you gauge if aging items are approaching your typical completion times.
        - A table will appear below the chart listing any items that are included in WIP but are not plotted as a dot.
        """)

        status_options = ["None"] + list(self.status_mapping.keys())
        sensible_done_options = [s for s in status_options if s.lower() == 'done']
        default_done_index = status_options.index(sensible_done_options[0]) if sensible_done_options else len(status_options) - 1

        with st.expander("Work Item Age Chart Controls", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                self.selections["age_start_status"] = st.selectbox("Start Status for Age Calculation", status_options, key="age_start", index=1 if len(status_options) > 1 else 0, help="Defines the start of the X-axis and the point from which item age is calculated.")
            with col2:
                try:
                    default_end_index = status_options.index("In Testing")
                except ValueError:
                    default_end_index = len(status_options) - 2 if len(status_options) > 2 else 0
                self.selections["age_end_status"] = st.selectbox("End Status for Axis", status_options, key="age_end", index=default_end_index, help="Defines the end of the X-axis.")
            with col3:
                self.selections["age_true_final_status"] = st.selectbox("Select the true 'Done' status", status_options, key="age_final", index=default_done_index, help="Select the status that marks an item as completely finished for your workflow.")

        self.selections["age_start_col"] = self.status_mapping.get(self.selections["age_start_status"])
        self.selections["age_end_col"] = self.status_mapping.get(self.selections["age_end_status"])
        self.selections["age_true_final_col"] = self.status_mapping.get(self.selections["age_true_final_status"])
        
        st.divider()

        if not all([self.selections["age_start_col"], self.selections["age_end_col"], self.selections["age_true_final_col"]]):
            st.info("To see the chart, please select a 'Start Status for Age Calculation', an 'End Status for Axis', and a true 'Done' status from the controls above.")
            return

        try:
            raw_cols = list(self.raw_df.columns)
            start_idx = raw_cols.index(self.selections["age_start_col"])
            end_idx = raw_cols.index(self.selections["age_end_col"])
            status_order = [s.replace("'->", "").strip() for s in raw_cols[start_idx : end_idx + 1]]
        except (ValueError, IndexError):
            st.error("Could not determine status order for the chart axis. Check your axis status selections.")
            return

        final_col = self.selections["age_true_final_col"]
        df_in_progress = self.raw_df[self.raw_df[final_col].apply(lambda x: pd.isna(DataProcessor._extract_latest_date(x)))].copy()

        df_for_wip_calc = df_in_progress[df_in_progress['Status'].isin(status_order)].copy()
        df_for_wip_calc = self._apply_all_filters(df_for_wip_calc, apply_date_filter=False)

        start_col = self.selections["age_start_col"]
        df_for_plotting = df_for_wip_calc.copy()
        df_for_plotting['Start date'] = df_for_plotting[start_col].apply(DataProcessor._extract_latest_date)
        df_for_plotting.dropna(subset=['Start date'], inplace=True)
        
        cycle_stats = StatsCalculator.cycle_time_stats(self.filtered_df) if self.filtered_df is not None and not self.filtered_df.empty else None

        chart = ChartGenerator.create_work_item_age_chart(
            plot_df=df_for_plotting,
            wip_df=df_for_wip_calc,
            status_order=status_order,
            cycle_time_percentiles=cycle_stats or {},
            percentile_settings=self.selections["percentiles"],
            is_color_blind_mode=self.selections['color_blind_mode']
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No work items currently in progress were found based on your status selections.")

        plotted_keys = df_for_plotting['Key'].tolist()
        unplotted_df = df_for_wip_calc[~df_for_wip_calc['Key'].isin(plotted_keys)]

        if not unplotted_df.empty:
            expander_title = f"#### View {len(unplotted_df)} item(s) counted in WIP but not aged from '{self.selections['age_start_status']}'"
            with st.expander(expander_title):
                st.markdown(f"The following items are included in the WIP counts above but do not have a start date in the **'{self.selections['age_start_status']}'** column, so their age cannot be plotted from that point. Their full date history is shown below.")

                display_cols = ['Key', 'Work type', 'Status'] + list(self.status_mapping.values())
                display_df = self.raw_df[self.raw_df['Key'].isin(unplotted_df['Key'])][display_cols]

                st.dataframe(display_df)


    def _display_wip_chart(self):
        """Displays the WIP chart."""
        st.header("Work In Progress (WIP) Trend")

        status_options = ["None"] + list(self.status_mapping.keys())
        col1, col2 = st.columns(2)

        with col1:
            self.selections["wip_start_status"] = st.selectbox("WIP Start Status", status_options, key="wip_start")
        with col2:
            self.selections["wip_done_status"] = st.selectbox("WIP End Status", status_options, key="wip_end")

        self.selections["wip_start_col"] = self.status_mapping.get(self.selections["wip_start_status"])
        self.selections["wip_done_col"] = self.status_mapping.get(self.selections["wip_done_status"])

        st.divider()

        if not self.selections["wip_start_col"] or not self.selections["wip_done_col"]:
            st.info("ℹ️ Please select a Start and End Status above to generate the WIP chart.")
            return

        wip_processed_df = DataProcessor.process_dates(self.raw_df, self.selections["wip_start_col"], self.selections["wip_done_col"])
        source_df = self._apply_all_filters(wip_processed_df, apply_date_filter=False)

        chart = ChartGenerator.create_wip_chart(source_df, self.selections['date_range'], self.selections['custom_start_date'], self.selections['custom_end_date'])
        if chart: st.plotly_chart(chart, use_container_width=True)
        else: st.warning("⚠️ No items with start dates for WIP chart.")

    def _display_throughput_chart(self):
        """Displays the Throughput chart and its controls."""
        st.header("Throughput")
        st.markdown('<div class="styled-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
            - **What it is:** This chart shows the number of work items completed per unit of time (day, week, or fortnight).
            - **How to read it:** Each bar represents a time period, and its height shows the number of items that were completed in that period.
            - **What to look for:**
                - **Consistency:** A relatively consistent throughput over time indicates a stable and predictable process.
                - **Variability:** High variability (large spikes and drops) can suggest that work items are not "right-sized" or that the team is being affected by outside interruptions or dependencies.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("Throughput measures the number of work items completed per unit of time. Use the control below to change the time unit.")

        col1, col2, col3 = st.columns(3)
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

        self.selections['sprint_anchor_date'] = None
        if self.selections["throughput_interval"] == 'Fortnightly':
            with col3:
                self.selections['sprint_anchor_date'] = st.date_input("Sprint End Date", value=datetime.now(), help="Select the last day of any Sprint in your team's cadence.")

        self.selections['throughput_status_col'] = self.status_mapping.get(self.selections["throughput_status"])

        st.divider()

        source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        
        max_date_df = self.raw_df.copy()
        if self.selections['throughput_status_col']:
            max_date_df['Throughput Date'] = max_date_df[self.selections['throughput_status_col']].apply(DataProcessor._extract_latest_date)
            max_date_df.dropna(subset=['Throughput Date'], inplace=True)
            overall_max_date = max_date_df['Throughput Date'].max() if not max_date_df.empty else None
        else:
            overall_max_date = None


        chart, agg_df = ChartGenerator.create_throughput_chart(
            source_df,
            self.selections["throughput_interval"],
            self.selections['throughput_status_col'],
            self.selections['date_range'],
            self.selections['custom_start_date'],
            self.selections['custom_end_date'],
            sprint_anchor_date=self.selections.get('sprint_anchor_date'),
            overall_max_date=overall_max_date
        )
        
        if chart:
            if agg_df is not None and len(agg_df) < 10:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ No items with the selected throughput status for this chart.")

    def _display_forecast_charts(self):
        """Displays the Forecasting charts and controls."""
        st.header("Throughput Forecasting")
        st.markdown('<div class="styled-expander">', unsafe_allow_html=True)
        with st.expander("Learn more about this chart", icon="🎓"):
            st.markdown("""
            - **What it is:** These charts use a Monte Carlo simulation to forecast future outcomes based on your team's historical throughput data.
            - **How to read it:** The charts run thousands of simulations of your future work to generate a range of possible outcomes and the probability of achieving them. For example, a result might say "There is an 85% chance of completing 12 or more items in the next two weeks."
            - **Why use Monte Carlo?** Traditional forecasting uses simple averages, which can be misleading and hide risk (this is known as the "Flaw of Averages"). A Monte Carlo simulation is a more robust statistical method that accounts for the variability in your past performance. By running thousands of simulations, it provides a much more realistic and trustworthy range of future outcomes and the probabilities associated with them.
            - **A Note on "Right-Sizing":** Forecasts are most reliable when the work items are "right-sized." Based on the guidance from industry experts, this means each item should be broken down into the smallest possible chunk that still delivers value and can be completed within your team's Service Level Expectation (SLE).
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        throughput_status = self.selections.get('throughput_status')
        if not throughput_status or throughput_status == "None":
            st.info("To run a forecast, first go to the 'Throughput' tab and choose the status that represents work being completed.")
            return

        info_text = (
            "Uses historical throughput data to run Monte Carlo simulations and forecast future outcomes. "
            f"The forecast simulation is based on items reaching the **'{throughput_status}'** status selected on the Throughput chart."
        )
        st.info(info_text)
        
        forecast_source_df = self._apply_all_filters(self.raw_df, apply_date_filter=False)
        throughput_status_col = self.selections.get('throughput_status_col')
        
        # --- Stability Check ---
        weekly_throughput, _ = ChartGenerator._get_recent_weekly_throughput(forecast_source_df, throughput_status_col)
        
        with st.expander("Data Stability Check"):
            if weekly_throughput is None or len(weekly_throughput) < 4:
                st.warning("Not enough historical data to perform a stability check. At least 4 weeks of throughput data are needed.")
            else:
                self._display_stability_check(weekly_throughput)

        forecast_tabs = st.tabs(["**How Many** (by date)", "**When** (by # of items)"])

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

            range_selection = self.selections.get("forecast_range")
            if range_selection == "Next 30 days": forecast_days = 30
            elif range_selection == "Next 60 days": forecast_days = 60
            elif range_selection == "Next 90 days": forecast_days = 90
            elif range_selection == "Custom":
                custom_date = self.selections.get("forecast_custom_date")
                if custom_date:
                    delta = (custom_date - datetime.now().date()).days
                    forecast_days = max(1, delta)
                else: forecast_days = 30
            else: forecast_days = 30

            chart = ChartGenerator.create_how_many_forecast_chart(forecast_source_df, forecast_days, throughput_status_col, self.selections['color_blind_mode'])

            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient historical data for forecasting. Check that the selected Throughput Status has completed items.")

        with forecast_tabs[1]:
            st.subheader("When can we expect to complete a given number of items?")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                items_to_complete = st.number_input("Number of items to forecast:", min_value=1, value=20, step=1, key="when_forecast_items")
            with col2:
                scope_complexity = st.selectbox("Scope Complexity", options=['Clear and understood', 'Somewhat understood', 'Not really understood yet', 'Very unclear or not understood'], key="scope_complexity")
            with col3:
                team_focus = st.selectbox("Team Focus", options=['100% (only this work)', '75% (mostly this work)', '50% (half of this work)', '25% (some of this work)'], key="team_focus")
            with col4:
                forecast_start_date = st.date_input("Forecast start date", value=datetime.now().date(), key="when_forecast_start")

            st.divider()

            chart, stats = ChartGenerator.create_when_forecast_chart(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col, scope_complexity, team_focus, self.selections['color_blind_mode'])

            if stats:
                st.markdown("""
                <style>
                .forecast-box {
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                    text-align: center;
                    margin-bottom: 10px;
                }
                .forecast-label { font-size: 1.1em; font-weight: bold; }
                .forecast-value { font-size: 2em; font-weight: bold; }
                </style>
                """, unsafe_allow_html=True)

                box_colors = ColorManager.get_forecast_box_colors(self.selections['color_blind_mode'])
                text_color = "#212529"

                cols = st.columns(len(stats))
                for i, (p, date_val) in enumerate(stats.items()):
                    with cols[i]:
                        background_color = box_colors.get(p, '#e9ecef')
                        st.markdown(f"""
                        <div class="forecast-box" style="background-color: {background_color}; color: {text_color};">
                            <div class="forecast-label">{p}% Likelihood</div>
                            <div class="forecast-value">{date_val.strftime("%d %b, %Y")}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient historical data for forecasting. Check that the selected Throughput Status has completed items.")

            with st.expander("🔍 Explore Forecast Scenarios"):
                st.markdown("This section explores how your completion date changes depending on whether your first week is 'good' (above median) or 'bad' (at or below median).")
                scenario_stats = ChartGenerator.run_when_scenario_forecast(forecast_source_df, items_to_complete, forecast_start_date, throughput_status_col)
                if scenario_stats and 'median' in scenario_stats:
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

    def _display_stability_check(self, throughput_data: pd.Series):
        """Calculates and displays the stability check metrics."""
        shuffled_data = throughput_data.sample(frac=1).reset_index(drop=True)
        split_point = len(shuffled_data) // 2
        
        group1 = shuffled_data.iloc[:split_point]
        group2 = shuffled_data.iloc[split_point:]

        avg1, avg2 = group1.mean(), group2.mean()
        median1, median2 = group1.median(), group2.median()

        if avg1 == 0 and avg2 == 0:
            stability_score = 0
        else:
            stability_score = abs(avg1 - avg2) / ((avg1 + avg2) / 2) * 100

        if stability_score < 15:
            score_category = "Stable"
            message = f"✅ **Stable Data ({stability_score:.1f}% variation):** Your historical data is consistent and well-suited for forecasting."
        elif stability_score < 30:
            score_category = "Some Variability"
            message = f"⚠️ **Some Variability ({stability_score:.1f}% variation):** Your data shows some inconsistencies. The forecast is still useful but should be considered with this in mind."
        else:
            score_category = "High Variability"
            message = f"🚨 **High Variability ({stability_score:.1f}% variation):** Your process appears unstable. Forecasts based on this data may be unreliable and should be used with caution."
            
        st.markdown(f'<p style="color:{ColorManager.STABILITY_SCORE_COLORS[score_category]}; font-weight: bold;">{message}</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Group 1 (Random Half)")
            st.metric("Average Throughput", f"{avg1:.2f}")
            st.metric("Median Throughput", f"{median1:.2f}")
        with col2:
            st.markdown("##### Group 2 (Random Half)")
            st.metric("Average Throughput", f"{avg2:.2f}")
            st.metric("Median Throughput", f"{median2:.2f}")
            
        if score_category == "High Variability":
            st.markdown("""
            ---
            **Potential Causes of High Variability:**
            - **Work Item Size:** A mix of very large and very small items being completed in different weeks.
            - **Team Availability:** Holidays or sick leave impacting some weeks more than others.
            - **Batching of Work:** Closing many items at once (e.g., at the end of a sprint) instead of a steady flow.
            - **External Blockers:** Periods of waiting for other teams, systems, or information.
            - **Process or Team Changes:** Recent changes to your team's structure or way of working.
            
            This stability check is a prompt to investigate *why* your process is unpredictable to improve your flow.
            """)


def display_welcome_message():
    """Displays the initial welcome message and instructions."""
    st.markdown("### 👋 Welcome to the Flow Metrics Dashboard")
    
    def _create_inline_link_with_logo(text: str, logo_path: str, url: str) -> str:
        """Creates a markdown-compatible HTML string for a link with an inline logo."""
        try:
            # Check if the logo file exists before trying to open it
            if not os.path.exists(logo_path):
                return f'<a href="{url}" target="_blank">**{text}**</a>'
            
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            return (
                f'<a href="{url}" target="_blank" style="text-decoration: none; color: inherit; font-weight: bold;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="height: 1.1em; vertical-align: -0.2em; margin-right: 5px;">'
                f'{text}</a>'
            )
        except Exception: # Broad exception to catch any file handling errors
            return f'<a href="{url}" target="_blank">**{text}**</a>'
            
    pro_url = "https://marketplace.atlassian.com/apps/1221826/status-time-reports-time-in-status"
    pro_link = _create_inline_link_with_logo("Status Time Reports", "Status time pro icon.png", pro_url)
    
    free_url = "https://marketplace.atlassian.com/apps/1222051/status-time-reports-free-time-in-status?hosting=cloud&tab=overview"
    free_link = _create_inline_link_with_logo("Status Time Reports Free", "Status time free icon.png", free_url)

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
    st.success(
        "**Your data is safe.** This application processes your CSV file entirely within your browser. "
        "No data from your uploaded file is ever sent to, saved, or stored on any server. When you close this browser tab, your data is permanently discarded."
    )

def _apply_date_filter(df: pd.DataFrame, date_col_name: str, date_range: str, custom_start_date, custom_end_date) -> pd.DataFrame:
    """Filters a DataFrame based on a date column and a selected date range string."""
    if date_range == "All time" or pd.to_datetime(df[date_col_name], errors='coerce').isna().all():
        return df
    
    today = pd.to_datetime(datetime.now().date())
    if date_range == "Last 30 days":
        cutoff = today - pd.DateOffset(days=30)
        return df[(df[date_col_name] >= cutoff) & (df[date_col_name] <= today)]
    elif date_range == "Last 60 days":
        cutoff = today - pd.DateOffset(days=60)
        return df[(df[date_col_name] >= cutoff) & (df[date_col_name] <= today)]
    elif date_range == "Last 90 days":
        cutoff = today - pd.DateOffset(days=90)
        return df[(df[date_col_name] >= cutoff) & (df[date_col_name] <= today)]
    elif date_range == "Custom" and custom_start_date and custom_end_date:
        start_date = pd.to_datetime(custom_start_date)
        end_date = pd.to_datetime(custom_end_date)
        return df[(df[date_col_name] >= start_date) & (df[date_col_name] <= end_date)]
    return df

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