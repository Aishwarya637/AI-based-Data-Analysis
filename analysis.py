import io
import os
from dotenv import load_dotenv
from google import genai
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Load environment variables from .env file
load_dotenv()

# 2. Initialize Gemini API Client
try:
  client = genai.Client()
except Exception as e:
  st.error(f"Failed to initialize Gemini Client. Check your .env file: {e}")

# Page Configuration
st.set_page_config(
    page_title="AI Auto Data Analyst & Dashboard", page_icon="📊", layout="wide"
)

st.title("📊 AI-Powered Automated Data Analysis & Dashboard")
st.write(
    "Upload your dataset (CSV) in the sidebar. The AI will analyze the data"
    " structure, provide executive insights, and let you switch between "
    "interactive visualization charts!"
)

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV data file", type=["csv"]
)

if uploaded_file is not None:
  # Load dataset using pandas
  df = pd.read_csv(uploaded_file)

  st.subheader("📁 Dataset Preview")
  st.dataframe(df.head())

  # Display high-level metrics
  col1, col2, col3 = st.columns(3)
  col1.metric("Total Rows", df.shape[0])
  col2.metric("Total Columns", df.shape[1])
  col3.metric("Missing Values", df.isnull().sum().sum())

  # Trigger AI Analysis
  if st.button("🚀 Run AI Data Analysis"):
    with st.spinner(
        "Gemini is analyzing patterns and trends in your dataset..."
    ):
      # Capture dataframe structure info into a string buffer safely using io.StringIO
      buffer = io.StringIO()
      df.info(buf=buffer)
      df_info_str = buffer.getvalue()

      prompt = f"""
            You are an expert Data Scientist. Analyze this dataset summary and give me:
            1. 3-4 key business insights or data trends.
            2. Potential data quality notes or anomalies to watch out for.
            
            Dataset Info:
            {df_info_str}
            
            First 5 Rows:
            {df.head().to_string()}
            
            Statistical Summary:
            {df.describe().to_string()}
            """

      try:
        # Call Gemini model using the updated active model name
        response = client.models.generate_content(
            model="gemini-3.8-flash", contents=prompt
        )

        st.subheader("🧠 AI Executive Summary & Insights")
        st.markdown(response.text)

      except Exception as e:
        st.error(f"Error generating AI analysis: {e}")

  # --- Automated Interactive Dashboard Section with Selectbox ---
  st.markdown("---")
  st.subheader("📈 Automated Interactive Dashboard")

  numeric_cols = df.select_dtypes(
      include=["int64", "float64"]
  ).columns.tolist()
  categorical_cols = df.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()
  all_cols = df.columns.tolist()

  if len(all_cols) > 0:
    # Selectbox for chart types (Default selection: Bar Chart)
    chart_options = [
        "Bar Chart",
        "Line Chart",
        "Pie Chart",
        "Histogram",
        "Scatter Plot",
    ]
    selected_chart = st.selectbox(
        "Select Visualization Type", chart_options, index=0
    )

    st.markdown(f"### Displaying: {selected_chart}")

    # Dynamic chart rendering based on selection
    if selected_chart == "Bar Chart":
      c1, c2 = st.columns(2)
      with c1:
        x_col = st.selectbox("X-Axis (Category)", all_cols, index=0)
      with c2:
        y_col = st.selectbox(
            "Y-Axis (Value)", numeric_cols if numeric_cols else all_cols
        )

      if df[x_col].duplicated().any() and y_col in numeric_cols:
        plot_df = df.groupby(x_col)[y_col].mean().reset_index().head(20)
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            title=f"Average {y_col} by {x_col}",
            color=x_col,
        )
      else:
        fig = px.bar(
            df.head(20),
            x=x_col,
            y=y_col,
            title=f"{y_col} by {x_col}",
            color=x_col if x_col in categorical_cols else None,
        )
      st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == "Line Chart":
      c1, c2 = st.columns(2)
      with c1:
        x_col = st.selectbox("X-Axis", all_cols, index=0)
      with c2:
        y_col = st.selectbox(
            "Y-Axis", numeric_cols if numeric_cols else all_cols
        )

      fig = px.line(
          df.head(50),
          x=x_col,
          y=y_col,
          markers=True,
          title=f"Line Chart: {y_col} over {x_col}",
      )
      st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == "Pie Chart":
      c1, c2 = st.columns(2)
      with c1:
        names_col = st.selectbox(
            "Names / Categories",
            categorical_cols if categorical_cols else all_cols,
        )
      with c2:
        values_col = st.selectbox(
            "Values (Numeric)", numeric_cols if numeric_cols else all_cols
        )

      pie_df = df.groupby(names_col)[values_col].sum().reset_index().head(10)
      fig = px.pie(
          pie_df,
          names=names_col,
          values=values_col,
          title=f"Pie Chart of {values_col} by {names_col}",
      )
      st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == "Histogram":
      num_col = st.selectbox(
          "Select Metric for Distribution",
          numeric_cols if numeric_cols else all_cols,
      )
      fig = px.histogram(
          df,
          x=num_col,
          title=f"Distribution of {num_col}",
          color_discrete_sequence=["#4C78A8"],
      )
      st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == "Scatter Plot":
      if len(numeric_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1:
          x_axis = st.selectbox("Scatter Plot X-Axis", numeric_cols, index=0)
        with c2:
          y_axis = st.selectbox(
              "Scatter Plot Y-Axis",
              numeric_cols,
              index=min(1, len(numeric_cols) - 1),
          )
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs {x_axis}",
            color_discrete_sequence=["#F58518"],
        )
        st.plotly_chart(fig, use_container_width=True)
      else:
        st.info("Dataset requires at least 2 numeric columns for a scatter plot.")
  else:
    st.warning("No columns found in the dataset.")

else:
  st.info("👈 Please upload a CSV dataset file using the sidebar to begin.")