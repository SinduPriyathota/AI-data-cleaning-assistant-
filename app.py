
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #172554);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main heading */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 5px;
}

.gradient-text {
    color: #6d4aff;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

/* Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
    min-height: 125px;
}

.metric-title {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
}

.metric-value {
    color: #172554;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

/* Information cards */
.info-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
}

/* Upload card */
.upload-card {
    background: white;
    border: 2px dashed #8b5cf6;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px 0 10px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = ""


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_quality_score(df):

    if df is None or df.empty:
        return 0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_penalty = (missing_cells / total_cells) * 50
    duplicate_penalty = (duplicate_rows / len(df)) * 30

    score = 100 - missing_penalty - duplicate_penalty

    return round(max(0, min(100, score)), 1)


def detect_outliers(df):

    outlier_counts = {}

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        values = df[column].dropna()

        if values.empty:
            outlier_counts[column] = 0
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            outlier_counts[column] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (df[column] < lower) |
            (df[column] > upper)
        ).sum()

        outlier_counts[column] = int(count)

    return outlier_counts


def clean_data(df):

    cleaned = df.copy()

    # Remove duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Clean column names
    cleaned.columns = (
        cleaned.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Fill missing values
    for column in cleaned.columns:

        if cleaned[column].isnull().sum() > 0:

            if pd.api.types.is_numeric_dtype(
                cleaned[column]
            ):

                median_value = cleaned[column].median()

                if pd.isna(median_value):
                    median_value = 0

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

            else:

                mode_value = cleaned[column].mode()

                if not mode_value.empty:

                    cleaned[column] = (
                        cleaned[column]
                        .fillna(mode_value.iloc[0])
                    )

                else:

                    cleaned[column] = (
                        cleaned[column]
                        .fillna("Unknown")
                    )

    # Remove unnecessary spaces from text
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:

        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
        )

    return cleaned


def create_sample_data():

    data = {
        "Employee_ID": [
            101, 102, 103, 104,
            105, 106, 107, 107
        ],

        "Name": [
            "Aarav Sharma",
            "Neha Verma",
            "Rohan Patel",
            "Priya Singh",
            "Karan Mehta",
            "Ananya Rao",
            "Rahul Kumar",
            "Rahul Kumar"
        ],

        "Age": [
            28, 32, 25, np.nan,
            34, 29, 42, 42
        ],

        "Department": [
            "IT",
            "HR",
            "Finance",
            "Marketing",
            "IT",
            "Sales",
            "IT",
            "IT"
        ],

        "Salary": [
            50000,
            62000,
            45000,
            58000,
            np.nan,
            55000,
            120000,
            120000
        ],

        "Performance_Score": [
            85,
            92,
            78,
            88,
            95,
            np.nan,
            110,
            110
        ]
    }

    return pd.DataFrame(data)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🗄️ AI Data")
    st.markdown("### Cleaning Assistant")

    st.markdown("---")

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📤 Upload Data",
            "📊 Data Overview",
            "🧹 Cleaning Suggestions",
            "✨ Clean Data",
            "⬇️ Download"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    ### 💡 Clean Data. Better Decisions.

    ✓ Remove duplicates

    ✓ Handle missing values

    ✓ Detect outliers

    ✓ Improve data quality

    ✓ Download clean data
    """)

    st.markdown("---")

    st.caption("Built with Python + Streamlit")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        AI Data Cleaning
        <span class="gradient-text">Assistant</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Clean your data • Detect quality issues •
        Get analysis-ready datasets
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BACKGROUND / HERO IMAGE
# ============================================================

if os.path.exists("background.png"):

    st.image(
        "background.png",
        use_container_width=True
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    if st.session_state.df is None:

        st.markdown(
            "## 🚀 Welcome to your Data Cleaning Workspace"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="info-card">

            <h3>🧹 Intelligent Data Cleaning</h3>

            Upload your CSV or Excel dataset and
            automatically identify common data-quality problems.

            <br><br>

            <b>Detect:</b>

            <br><br>

            🔴 Missing Values

            <br>

            🔴 Duplicate Records

            <br>

            🟠 Numerical Outliers

            <br>

            🟣 Data Quality Issues

            <br>

            🟢 Overall Quality Score

            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown("""
            <div class="info-card">

            <h3>📋 Simple Workflow</h3>

            <br>

            <b>01</b> Upload your dataset

            <br><br>

            <b>02</b> Analyze data quality

            <br><br>

            <b>03</b> Review cleaning suggestions

            <br><br>

            <b>04</b> Clean your dataset

            <br><br>

            <b>05</b> Download cleaned data

            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🧪 Try Sample Data")

        if st.button(
            "✨ Load Sample Dataset",
            use_container_width=True
        ):

            st.session_state.df = create_sample_data()
            st.session_state.cleaned_df = None
            st.session_state.file_name = "sample_dataset.csv"

            st.success(
                "Sample dataset loaded successfully!"
            )

            st.rerun()

    else:

        df = st.session_state.df

        quality = calculate_quality_score(df)

        missing = int(
            df.isnull().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        outlier_counts = detect_outliers(df)

        total_outliers = sum(
            outlier_counts.values()
        )

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        TOTAL ROWS
                    </div>
                    <div class="metric-value">
                        {len(df):,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        TOTAL COLUMNS
                    </div>
                    <div class="metric-value">
                        {len(df.columns)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        MISSING VALUES
                    </div>
                    <div class="metric-value">
                        {missing:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        DUPLICATE ROWS
                    </div>
                    <div class="metric-value">
                        {duplicates:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        QUALITY SCORE
                    </div>
                    <div class="metric-value">
                        {quality}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("## 📊 Data Quality Overview")

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            values = [
                quality,
                100 - quality
            ]

            labels = [
                "Good Data",
                "Issues"
            ]

            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title(
                "Overall Data Quality"
            )

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            issue_data = pd.DataFrame({
                "Issue": [
                    "Missing Values",
                    "Duplicates",
                    "Outliers"
                ],
                "Count": [
                    missing,
                    duplicates,
                    total_outliers
                ]
            })

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            ax.bar(
                issue_data["Issue"],
                issue_data["Count"]
            )

            ax.set_title(
                "Detected Data Issues"
            )

            ax.set_ylabel("Count")

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "📤 Upload Data":

    st.markdown("## 📤 Upload Your Dataset")

    st.markdown("""
    <div class="upload-card">

    <h2>📁 Upload CSV or Excel File</h2>

    <p>
    Supported formats: CSV, XLSX and XLS
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.df = df
            st.session_state.cleaned_df = None
            st.session_state.file_name = (
                uploaded_file.name
            )

            st.success(
                f"✅ {uploaded_file.name} uploaded successfully!"
            )

            st.markdown("### 👀 Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Rows",
                    f"{df.shape[0]:,}"
                )

            with c2:
                st.metric(
                    "Columns",
                    df.shape[1]
                )

        except Exception as e:

            st.error(
                f"❌ Error reading file: {e}"
            )


# ============================================================
# DATA OVERVIEW
# ============================================================

elif page == "📊 Data Overview":

    if st.session_state.df is None:

        st.warning(
            "⚠️ Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        st.markdown("## 📊 Data Overview")

        st.markdown("### 👀 Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.markdown("### 🧾 Dataset Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with c2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with c3:

            memory = (
                df.memory_usage(
                    deep=True
                ).sum() / 1024
            )

            st.metric(
                "Memory",
                f"{memory:.1f} KB"
            )

        st.markdown("### 🧩 Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                int(
                    df[column].isnull().sum()
                )
                for column in df.columns
            ],
            "Unique Values": [
                int(
                    df[column].nunique()
                )
                for column in df.columns
            ]
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        st.markdown("### 📈 Statistical Summary")

        st.dataframe(
            df.describe(
                include="all"
            ).transpose(),
            use_container_width=True
        )


# ============================================================
# CLEANING SUGGESTIONS
# ============================================================

elif page == "🧹 Cleaning Suggestions":

    if st.session_state.df is None:

        st.warning(
            "⚠️ Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        st.markdown(
            "## 🧹 Cleaning Suggestions"
        )

        missing = int(
            df.isnull().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        outlier_data = detect_outliers(df)

        total_outliers = sum(
            outlier_data.values()
        )

        # Missing values
        if missing > 0:

            st.warning(
                f"""
                ⚠️ Missing Values Detected

                Your dataset contains
                {missing} missing values.

                Recommended action:
                fill numerical columns with median values
                and text columns with the most frequent value.
                """
            )

        else:

            st.success(
                "✅ No missing values detected."
            )

        # Duplicates
        if duplicates > 0:

            st.warning(
                f"""
                ⚠️ Duplicate Records Detected

                {duplicates} duplicate rows were found.

                Recommended action:
                remove duplicate records.
                """
            )

        else:

            st.success(
                "✅ No duplicate rows detected."
            )

        # Outliers
        if total_outliers > 0:

            st.warning(
                f"""
                ⚠️ Potential Outliers Detected

                {total_outliers} potential numerical
                outliers were found.

                Review them before removing them.
                """
            )

        else:

            st.success(
                "✅ No major numerical outliers detected."
            )

        st.markdown("### 🔎 Outlier Analysis")

        if outlier_data:

            outlier_df = pd.DataFrame(
                list(outlier_data.items()),
                columns=[
                    "Column",
                    "Outliers"
                ]
            )

            st.dataframe(
                outlier_df,
                use_container_width=True
            )

        st.markdown("### 💡 Recommended Cleaning Actions")

        recommendations = [
            "Remove duplicate rows",
            "Fill missing numerical values with median",
            "Fill missing text values with mode",
            "Remove unnecessary spaces",
            "Standardize column names",
            "Review numerical outliers"
        ]

        for recommendation in recommendations:

            st.write(
                f"✅ {recommendation}"
            )


# ============================================================
# CLEAN DATA
# ============================================================

elif page == "✨ Clean Data":

    if st.session_state.df is None:

        st.warning(
            "⚠️ Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        st.markdown(
            "## ✨ Clean Your Data"
        )

        st.info(
            """
            The automatic cleaning process will:

            • Remove duplicate rows

            • Fill missing numerical values

            • Fill missing text values

            • Clean column names

            • Remove unnecessary spaces
            """
        )

        if st.button(
            "🧹 Run Automatic Data Cleaning",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🔄 Analyzing and cleaning your dataset..."
            ):

                cleaned = clean_data(df)

                st.session_state.cleaned_df = cleaned

            st.success(
                "🎉 Data cleaning completed successfully!"
            )

        # Show results after cleaning
        if st.session_state.cleaned_df is not None:

            cleaned_df = (
                st.session_state.cleaned_df
            )

            st.markdown(
                "## 📊 Before vs After Cleaning"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🔴 Before Cleaning"
                )

                st.metric(
                    "Rows",
                    f"{df.shape[0]:,}"
                )

                st.metric(
                    "Missing Values",
                    f"{df.isnull().sum().sum():,}"
                )

                st.metric(
                    "Duplicate Rows",
                    f"{df.duplicated().sum():,}"
                )

            with col2:

                st.markdown(
                    "### 🟢 After Cleaning"
                )

                st.metric(
                    "Rows",
                    f"{cleaned_df.shape[0]:,}"
                )

                st.metric(
                    "Missing Values",
                    f"{cleaned_df.isnull().sum().sum():,}"
                )

                st.metric(
                    "Duplicate Rows",
                    f"{cleaned_df.duplicated().sum():,}"
                )

            st.markdown(
                "## 👀 Cleaned Dataset"
            )

            st.dataframe(
                cleaned_df,
                use_container_width=True
            )


# ============================================================
# DOWNLOAD CLEANED DATA
# ============================================================

elif page == "⬇️ Download":

    if st.session_state.cleaned_df is None:

        st.warning(
            "⚠️ Please clean your dataset first."
        )

    else:

        cleaned_df = (
            st.session_state.cleaned_df
        )

        st.markdown(
            "## 🎉 Data Cleaning Completed"
        )

        st.success(
            """
            Your dataset has been successfully cleaned
            and is ready to download!
            """
        )

        # Final metrics
        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Final Rows",
                f"{cleaned_df.shape[0]:,}"
            )

        with c2:

            st.metric(
                "Final Columns",
                cleaned_df.shape[1]
            )

        with c3:

            st.metric(
                "Missing Values",
                cleaned_df.isnull().sum().sum()
            )

        with c4:

            st.metric(
                "Duplicate Rows",
                cleaned_df.duplicated().sum()
            )

        st.markdown(
            "### 👀 Final Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )

        st.markdown(
            "## 📥 Download Your Cleaned Dataset"
        )

        st.write(
            "Choose your preferred format:"
        )

        # ----------------------------------------------------
        # CSV DOWNLOAD
        # ----------------------------------------------------

        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="AI_Cleaned_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.write("")

        # ----------------------------------------------------
        # EXCEL DOWNLOAD
        # ----------------------------------------------------

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            cleaned_df.to_excel(
                writer,
                index=False,
                sheet_name="Cleaned Data"
            )

        excel_data = (
            excel_buffer.getvalue()
        )

        st.download_button(
            label="📊 Download Cleaned Excel",
            data=excel_data,
            file_name="AI_Cleaned_Dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.markdown("---")

        st.markdown(
            """
            <div class="info-card">

            <h3>✨ Your data is now analysis-ready!</h3>

            You can use your cleaned dataset with:

            <br><br>

            📊 Power BI &nbsp;&nbsp;
            📈 Tableau &nbsp;&nbsp;
            📗 Excel &nbsp;&nbsp;
            🐍 Python &nbsp;&nbsp;
            🗄️ SQL &nbsp;&nbsp;
            🤖 Machine Learning

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI Data Cleaning Assistant |
        Python • Pandas • NumPy • Matplotlib • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

