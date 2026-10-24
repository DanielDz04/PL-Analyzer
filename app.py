
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Premier League Match Analyzer",
    page_icon="⚽",
    layout="wide"
)


st.set_page_config(
    page_title="Premier League Match Analyzer",
    page_icon="⚽",
    layout="wide"
)

# Application title
st.title("⚽ Premier League Match Analyzer")

# Application description
st.write("Explore Premier League statistics and predict match goals using linear regression.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Premier League CSV file", type=["csv"])

if uploaded_file is not None:

    # Read the dataset
    df = pd.read_csv(uploaded_file, encoding="cp1252")

    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show the complete dataset
    with st.expander("View Full Dataset"):
        st.dataframe(df)

    # Display dataset information
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    # Select numerical columns
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    st.subheader("Data Visualization")

    if len(numeric_columns) > 0:

        # Select chart type
        chart_type = st.selectbox(
            "Choose a chart type",
            ["Histogram", "Line Chart", "Scatter Plot"]
        )

        # Select numerical column
        column = st.selectbox(
            "Choose a numerical column",
            numeric_columns
        )

        # Create charts
        if chart_type == "Histogram":
            fig = px.histogram(df, x=column)

        elif chart_type == "Line Chart":
            fig = px.line(df, y=column)

        elif chart_type == "Scatter Plot":
            y_column = st.selectbox(
                "Choose a second numerical column",
                numeric_columns
            )
            fig = px.scatter(df, x=column, y=y_column)

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numerical columns found.")

    # Linear Regression
    st.subheader("Linear Regression Prediction")

    if len(numeric_columns) >= 3:

        # Select two input variables
        input1 = st.selectbox(
            "Choose the first input variable",
            numeric_columns,
            index=0
        )

        input2 = st.selectbox(
            "Choose the second input variable",
            numeric_columns,
            index=1
        )

        # Select target variable
        target = st.selectbox(
            "Choose the target variable",
            numeric_columns,
            index=2
        )

        if len({input1, input2, target}) < 3:
            st.warning("Please select three different variables.")

        else:
            # Remove missing values
            model_data = df[[input1, input2, target]].dropna()

            if len(model_data) < 5:
                st.warning("Not enough data to train the model.")

            else:
                # Define inputs and target
                X = model_data[[input1, input2]]
                y = model_data[target]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write("Mean Absolute Error:", round(mae, 2))
                st.write("R-squared Score:", round(r2, 2))

                # Enter values for prediction
                value1 = st.number_input(
                    "Enter a value for " + input1,
                    value=0.0
                )

                value2 = st.number_input(
                    "Enter a value for " + input2,
                    value=0.0
                )

                # Predict
                new_data = pd.DataFrame({
                    input1: [value1],
                    input2: [value2]
                })

                prediction = model.predict(new_data)

                st.success(
                    "Predicted " + target + ": " +
                    str(round(prediction[0], 2))
                )

                # Explain the model
                st.write(
                    "This model estimates " + target +
                    " using " + input1 + " and " + input2 +
                    ". Incorrect predictions may lead to inaccurate "
                    "conclusions about soccer match performance."
                )

    else:
        st.warning("At least three numerical columns are required.")
