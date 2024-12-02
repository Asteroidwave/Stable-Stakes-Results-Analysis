import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
file_name = "The Beginning GP.xlsx"  # Replace with your file name
data = pd.ExcelFile(file_name)
tier1_df = data.parse('Tier 1')  # Replace 'Tier 1' with your desired sheet name

# Convert 'Date' column to datetime format
if pd.api.types.is_numeric_dtype(tier1_df['Date']):
    tier1_df['Date'] = pd.to_datetime(tier1_df['Date'], origin='1899-12-30', unit='D')
else:
    tier1_df['Date'] = pd.to_datetime(tier1_df['Date'], errors='coerce')

tier1_df = tier1_df.dropna(subset=['Date'])  # Remove invalid dates


# Helper functions
def calculate_quartile_table(data):
    quartile_table = {}
    for col in ['Total Frequency', 'Total Average Odds']:
        q1 = data[col].quantile(0.25)
        q2 = data[col].quantile(0.50)
        q3 = data[col].quantile(0.75)
        quartile_table[col] = {
            "Q1 (25%)": q1,
            "Q2 (50%)": q2,
            "Q3 (75%)": q3,
            "Range": data[col].max() - data[col].min()
        }
    return pd.DataFrame(quartile_table)


def calculate_statistical_table(data):
    stats = {
        "Statistic": [
            "Count", "Min", "Max", "Mean", "Median", "Standard Deviation", "Range"
        ]
    }
    for col in ['Total Frequency', 'Total Average Odds', 'Total Points w M/x']:
        stats[col] = [
            data[col].count(),
            data[col].min(),
            data[col].max(),
            data[col].mean(),
            data[col].median(),
            data[col].std(),
            data[col].max() - data[col].min()
        ]
    return pd.DataFrame(stats)


# Sidebar for user input
st.sidebar.header("Filter Options")
unique_dates = ["Entire Sheet"] + tier1_df['Date'].dt.strftime('%m/%d/%Y').unique().tolist()
selected_date = st.sidebar.selectbox("Select Date", unique_dates)

# Range filters for Frequency, Odds, and Points
st.sidebar.subheader("Range Filters")
freq_min, freq_max = st.sidebar.slider("Frequency Range",
                                       int(tier1_df['Total Frequency'].min()),
                                       int(tier1_df['Total Frequency'].max()),
                                       (int(tier1_df['Total Frequency'].min()), int(tier1_df['Total Frequency'].max())))
odds_min, odds_max = st.sidebar.slider("Odds Range",
                                       float(tier1_df['Total Average Odds'].min()),
                                       float(tier1_df['Total Average Odds'].max()),
                                       (float(tier1_df['Total Average Odds'].min()),
                                        float(tier1_df['Total Average Odds'].max())))
points_min, points_max = st.sidebar.slider("Points Range",
                                           int(tier1_df['Total Points w M/x'].min()),
                                           int(tier1_df['Total Points w M/x'].max()),
                                           (int(tier1_df['Total Points w M/x'].min()),
                                            int(tier1_df['Total Points w M/x'].max())))

# Filter data based on selected date and range filters
if selected_date == "Entire Sheet":
    filtered_df = tier1_df
else:
    filtered_df = tier1_df[tier1_df['Date'] == pd.to_datetime(selected_date)]

filtered_df = filtered_df[(filtered_df['Total Frequency'].between(freq_min, freq_max)) &
                          (filtered_df['Total Average Odds'].between(odds_min, odds_max)) &
                          (filtered_df['Total Points w M/x'].between(points_min, points_max))]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Quartile Table
    st.subheader("Quartile Table")
    quartile_table = calculate_quartile_table(filtered_df)
    st.dataframe(quartile_table)

    # Basic Statistical Table
    st.subheader("Basic Statistical Table")
    statistical_table = calculate_statistical_table(filtered_df)
    st.dataframe(statistical_table)

    # Analysis Options
    analysis_type = st.sidebar.selectbox("Select Analysis Type", [
        "Linear Regression", "Outlier Analysis", "Efficiency Analysis",
        "Impact Analysis", "Correlation Analysis", "Predictive Modeling", "Time-Based Analysis"
    ])

    if analysis_type == "Linear Regression":
        st.subheader(f"Linear Regression Analysis ({selected_date})")
        factor = st.selectbox("Select Factor to Analyze", ["Total Frequency", "Total Average Odds", "Both"])
        if factor != "Both":
            X = filtered_df[[factor]]
            y = filtered_df['Total Points w M/x']
            model = LinearRegression().fit(X, y)
            coef, intercept = model.coef_[0], model.intercept_
            r2 = model.score(X, y) * 100

            fig = px.scatter(filtered_df, x=factor, y="Total Points w M/x",
                             title=f"Linear Regression: {factor} vs Total Points")
            fig.add_trace(
                go.Scatter(
                    x=filtered_df[factor],
                    y=model.predict(X),
                    mode='lines',
                    name='Regression Line',
                    line=dict(color="red"),
                    hovertemplate=f"Slope: {coef:.2f}, R²: {r2:.2f}%"
                )
            )
            st.plotly_chart(fig)
            st.write(f"""
                **Analysis**:  
                - Slope: {coef:.2f}  
                - Intercept: {intercept:.2f}  
                - R²: {r2:.2f}% (explains the variation in Total Points).
            """)
        else:
            # Both Total Frequency and Odds
            X = filtered_df[['Total Frequency', 'Total Average Odds']]
            y = filtered_df['Total Points w M/x']
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y) * 100

            fig = px.scatter_3d(filtered_df, x='Total Frequency', y='Total Average Odds', z='Total Points w M/x',
                                color='Total Points w M/x', title="Linear Regression (Frequency, Odds vs Points)")
            st.plotly_chart(fig)
            st.write(f"""
                **Analysis**:  
                The combined model explains **{r2:.2f}%** of the variance in Total Points.  
                This highlights how both frequency and odds interact to influence points.
            """)

    elif analysis_type == "Outlier Analysis":
        st.subheader("Outlier Analysis")
        col = st.radio("Select Feature", ["Total Frequency", "Total Average Odds", "Total Points w M/x", "All"])
        if col == "All":
            cols = ['Total Frequency', 'Total Average Odds', 'Total Points w M/x']
        else:
            cols = [col]
        fig = px.box(filtered_df, y=cols, title=f"Outlier Analysis: {col}")
        st.plotly_chart(fig)

    elif analysis_type == "Efficiency Analysis":
        st.subheader("Efficiency Analysis")
        filtered_df['Points per Frequency'] = filtered_df['Total Points w M/x'] / filtered_df['Total Frequency']
        filtered_df['Points per Multiplier'] = filtered_df['Total Points w M/x'] / filtered_df['Scoring Multiplier']

        efficiency = filtered_df[['Name', 'Points per Frequency', 'Points per Multiplier']].melt(id_vars='Name')
        fig = px.bar(efficiency, x='Name', y='value', color='variable', barmode='group',
                     labels={'value': 'Efficiency Value', 'variable': 'Efficiency Metric'},
                     title="Efficiency Analysis")
        st.plotly_chart(fig)
        st.write("**Analysis**: Efficiency metrics like points per frequency and multiplier provide insights "
                 "into the most valuable cards relative to their appearances or multipliers.")

    elif analysis_type == "Impact Analysis":
        st.subheader(f"Impact Analysis ({selected_date})")
        factor = st.radio("Select Impact Factor", ["Total Average Odds", "Total Frequency", "Scoring Multiplier"])
        increment = st.slider("Set Increment", 0.5, 5.0, 1.0)

        # Create bins and convert intervals to strings
        bins = np.arange(filtered_df[factor].min(), filtered_df[factor].max() + increment, increment)
        filtered_df['Range'] = pd.cut(filtered_df[factor], bins)
        filtered_df['Range'] = filtered_df['Range'].astype(str)  # Convert Interval to string for JSON serialization

        # Group data and calculate averages
        summary = filtered_df.groupby('Range').agg({
            'Total Points w M/x': 'mean',
            'Total Points no Mx': 'mean'
        }).reset_index()

        # Create bar chart for impact analysis
        fig = px.bar(summary, x='Range', y=['Total Points w M/x', 'Total Points no Mx'],
                     labels={'value': 'Points', 'variable': 'Metric'},
                     title=f"Impact Analysis: {factor}",
                     barmode='group')
        st.plotly_chart(fig)

        st.write(f"""
            **Analysis**:  
            - Factor: {factor}  
            - Increment: {increment}  
            - Highlights: Total Points (with and without multipliers) grouped by selected ranges.  
            - Use this to assess how variations in {factor} impact scoring performance.
        """)

    elif analysis_type == "Correlation Analysis":
        st.subheader(f"Correlation Analysis ({selected_date})")

        # Select factors to correlate with Total Points w M/x
        factors = st.multiselect("Select Factors to Correlate",
                                 ['Total Frequency', 'Total Average Odds', 'Scoring Multiplier'],
                                 default=['Total Frequency', 'Total Average Odds'])
        target = 'Total Points w M/x'

        if factors:
            # Compute correlation matrix
            corr_matrix = filtered_df[factors + [target]].corr()
            target_corr = corr_matrix[target].drop(target)  # Drop self-correlation

            # Bar chart for correlation coefficients
            fig = px.bar(target_corr, x=target_corr.index, y=target_corr.values,
                         labels={'x': 'Factors', 'y': 'Correlation Coefficient'},
                         title=f"Correlation with {target}")
            st.plotly_chart(fig)

            st.write(f"""
                **Analysis**:  
                - **Positive Correlation**: Factors that increase Total Points as they increase.  
                - **Negative Correlation**: Factors that decrease Total Points as they increase.  
                - Use this chart to identify which factors most strongly influence Total Points.
            """)
        else:
            st.warning("Please select at least one factor to correlate.")


    elif analysis_type == "Predictive Modeling":
        st.subheader(f"Predictive Modeling ({selected_date})")
        X = filtered_df[['Total Frequency', 'Total Average Odds', 'Scoring Multiplier']]
        y = filtered_df['Total Points w M/x']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Points', 'y': 'Predicted Points'},
                         title="Predictive Modeling: Actual vs Predicted Points")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig)

        st.write(f"""
            **Analysis**:  
            The predictive model explains **{model.score(X_test, y_test) * 100:.2f}%** of the variance in Total Points.  
            - **Key Features Used**: Total Frequency, Total Average Odds, Scoring Multiplier.  
            - **Insights**: The closer the R² value is to 100%, the more accurately the model predicts Total Points.
        """)

    elif analysis_type == "Time-Based Analysis":
        st.subheader(f"Time-Based Analysis ({selected_date})")

        time_analysis = filtered_df.groupby('Date').agg({
            'Total Frequency': 'sum',
            'Total Average Odds': 'mean',
            'Total Points w M/x': 'sum'
        }).reset_index()

        fig = px.line(time_analysis, x='Date', y=['Total Frequency', 'Total Average Odds', 'Total Points w M/x'],
                      labels={'value': 'Metric Value', 'variable': 'Metric'},
                      title="Time-Based Trends")
        st.plotly_chart(fig)

        st.write(f"""
            **Analysis**:  
            This visualization shows how metrics such as Frequency, Odds, and Points fluctuate over time.  
            - **Trends**: Identify patterns or anomalies in the data.  
            - **Insights**: Use this analysis to assess performance over the selected time range.
        """)



