
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Supply Chain Analytics Dashboard")
st.markdown(
    "Explore delivery performance, profitability, bottlenecks "
    "and root causes of delayed orders."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("DataCoSupplyChainDataset2.csv", encoding="latin1")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert numerical columns
    numeric_cols = [
        "Days for shipment (scheduled)",
        "Days for shipping (real)",
        "Order Profit Per Order",
        "Sales",
        "Late_delivery_risk"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate delivery delay
    if (
        "Days for shipping (real)" in df.columns
        and "Days for shipment (scheduled)" in df.columns
    ):
        df["Delay"] = (
            df["Days for shipping (real)"]
            - df["Days for shipment (scheduled)"]
        )

        df["Is_Delayed"] = (df["Delay"] > 0).astype(int)

    # Extract date-time features if not already present
    if "order date (DateOrders)" in df.columns:
        date = pd.to_datetime(
            df["order date (DateOrders)"],
            errors="coerce"
        )

        if "order_month" not in df.columns:
            df["order_month"] = date.dt.month

        if "order_day" not in df.columns:
            df["order_day"] = date.dt.day_name()

        if "order_hour" not in df.columns:
            df["order_hour"] = date.dt.hour

    return df


df = load_data()

# --------------------------------------------------
# BASIC VALIDATION
# --------------------------------------------------

required_cols = [
    "Delay",
    "Is_Delayed",
    "Order Region",
    "Customer Segment",
    "Shipping Mode",
    "Order Status",
    "Type",
    "Department Name"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Required columns missing: {missing_cols}")
    st.stop()

# --------------------------------------------------
# TOP FILTERS
# --------------------------------------------------

st.markdown("---")
st.subheader("🔎 Filter Your Analysis")

filter_columns = [
    "Order Region",
    "Customer Segment",
    "Shipping Mode",
    "Department Name",
    "Type",
    "Order Status"
]

selected_filters = {}

filter_cols = st.columns(3)

for i, col in enumerate(filter_columns):

    options = sorted(
        df[col].dropna().astype(str).unique().tolist()
    )

    with filter_cols[i % 3]:
        selected_filters[col] = st.selectbox(
            col,
            ["All"] + options,
            key=f"filter_{col}"
        )

# Apply all filters
filtered_df = df.copy()

for col, value in selected_filters.items():

    if value != "All":
        filtered_df = filtered_df[
            filtered_df[col].astype(str) == value
        ]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

st.caption(
    f"Showing {len(filtered_df):,} orders out of {len(df):,} total records."
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

primary_color = "#2563EB"
accent_color = "#F59E0B"
danger_color = "#DC2626"

sns.set_theme(style="whitegrid")


def show_chart(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def format_profit(value, tick_number):
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:.0f}"


def delay_percentage(data, category):

    result = (
        data.groupby(category)
        .agg(
            total_orders=("Delay", "count"),
            late_orders=("Is_Delayed", "sum")
        )
        .reset_index()
    )

    result["delay_pct"] = (
        result["late_orders"] / result["total_orders"] * 100
    )

    return result.sort_values(
        "delay_pct", ascending=False
    ).head(10)


# --------------------------------------------------
# SECTION 1: BASIC SUPPLY CHAIN ANALYSIS
# 3 CHARTS
# --------------------------------------------------

st.markdown("---")
st.header("1. Basic Supply Chain Analysis")

# Chart 1: Delivery Risk Distribution

st.subheader("1.1 Delivery Risk Distribution")

if "Late_delivery_risk" in filtered_df.columns:

    risk_counts = filtered_df["Late_delivery_risk"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [
        "High Risk" if x == 1 else "Low Risk"
        for x in risk_counts.index
    ]

    ax.pie(
        risk_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )

    ax.set_title("Distribution of Late Delivery Risk")

    show_chart(fig)

else:
    st.warning("Late_delivery_risk column not found.")


# Chart 2: Orders by Shipping Mode

st.subheader("1.2 Orders by Shipping Mode")

shipping_counts = (
    filtered_df["Shipping Mode"]
    .value_counts()
    .reset_index()
)

shipping_counts.columns = ["Shipping Mode", "Orders"]

fig, ax = plt.subplots(figsize=(9, 4))

sns.barplot(
    data=shipping_counts,
    x="Shipping Mode",
    y="Orders",
    color=primary_color,
    ax=ax
)

ax.set_title("Order Distribution by Shipping Mode")
ax.set_xlabel("Shipping Mode")
ax.set_ylabel("Number of Orders")

plt.xticks(rotation=20)
show_chart(fig)


# Chart 3: Orders by Region

st.subheader("1.3 Orders by Region")

region_counts = (
    filtered_df["Order Region"]
    .value_counts()
    .head(10)
    .reset_index()
)

region_counts.columns = ["Order Region", "Orders"]

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=region_counts,
    x="Orders",
    y="Order Region",
    color=accent_color,
    ax=ax
)

ax.set_title("Top 10 Regions by Order Volume")
ax.set_xlabel("Number of Orders")
ax.set_ylabel("Order Region")

show_chart(fig)


# --------------------------------------------------
# SECTION 2: PROFITABILITY VS DELIVERY TIME
# 2 CHARTS
# --------------------------------------------------

st.markdown("---")
st.header("2. Profitability vs Delivery Time Analysis")

if "Order Profit Per Order" in filtered_df.columns:

    profit_df = filtered_df.dropna(
        subset=["Delay", "Order Profit Per Order"]
    ).copy()

    # Chart 4: Delay Distribution

    st.subheader("2.1 Delay Distribution")

    delay_distribution = (
        profit_df["Delay"]
        .value_counts(normalize=True)
        .mul(100)
        .sort_index()
        .reset_index()
    )

    delay_distribution.columns = [
        "Delay_Days",
        "Percentage"
    ]

    # Limit extreme delay values for readability
    delay_plot = delay_distribution[
        delay_distribution["Delay_Days"].between(
            delay_distribution["Delay_Days"].quantile(0.01),
            delay_distribution["Delay_Days"].quantile(0.99)
        )
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.barplot(
        data=delay_plot,
        x="Delay_Days",
        y="Percentage",
        color=accent_color,
        ax=ax
    )

    ax.set_title("Distribution of Delivery Delay Days")
    ax.set_xlabel("Delay (Days)")
    ax.set_ylabel("Percentage of Orders (%)")

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{height:.1f}%",
            ha="center",
            fontsize=8
        )

    show_chart(fig)

    # Chart 5: Profit Analysis by Delay Days

    st.subheader("2.2 Profit Analysis by Delay Days")

    profit_metrics = (
        profit_df.groupby("Delay")
        .agg(
            total_profit=("Order Profit Per Order", "sum"),
            mean_profit=("Order Profit Per Order", "mean"),
            total_orders=("Order Profit Per Order", "count")
        )
        .reset_index()
        .sort_values("Delay")
    )

    profit_plot = profit_metrics[
        profit_metrics["Delay"].between(
            profit_metrics["Delay"].quantile(0.01),
            profit_metrics["Delay"].quantile(0.99)
        )
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Profit Metrics by Delay Day")
        st.dataframe(
            profit_metrics.round(2),
            use_container_width=True,
            height=300
        )

    with col2:

        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.bar(
            profit_plot["Delay"],
            profit_plot["total_profit"],
            color=primary_color
        )

        ax1.set_xlabel("Delay Days")
        ax1.set_ylabel("Total Profit")
        ax1.set_title("Total Profit and Mean Profit by Delay")

        ax1.yaxis.set_major_formatter(
            FuncFormatter(format_profit)
        )

        ax2 = ax1.twinx()

        ax2.plot(
            profit_plot["Delay"],
            profit_plot["mean_profit"],
            marker="o",
            color=accent_color,
            linewidth=2
        )

        ax2.set_ylabel("Mean Profit")

        show_chart(fig)

else:
    st.warning("Order Profit Per Order column not found.")


# --------------------------------------------------
# SECTION 3: BOTTLENECK DETECTION
# 6 CHARTS
# --------------------------------------------------

st.markdown("---")
st.header("3. Bottleneck Detection")

st.write(
    "Identify the regions, customer segments, shipping modes "
    "and departments with the highest delivery delay percentage."
)

categories = [
    "Order Region",
    "Customer Segment",
    "Shipping Mode",
    "Order Status",
    "Type",
    "Department Name"
]

chart_titles = [
    "3.1 Delay % by Order Region",
    "3.2 Delay % by Customer Segment",
    "3.3 Delay % by Shipping Mode",
    "3.4 Delay % by Order Status",
    "3.5 Delay % by Type",
    "3.6 Delay % by Department Name"
]

for i, category in enumerate(categories):

    st.subheader(chart_titles[i])

    cat_df = delay_percentage(filtered_df, category)

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        data=cat_df,
        x="delay_pct",
        y=category,
        color=primary_color,
        ax=ax
    )

    ax.set_title(f"Delay Percentage by {category}")
    ax.set_xlabel("Delay Percentage (%)")
    ax.set_ylabel(category)

    for bar in ax.patches:
        width = bar.get_width()

        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=9
        )

    max_value = cat_df["delay_pct"].max()

    if pd.notna(max_value):
        ax.set_xlim(0, max_value * 1.18)

    show_chart(fig)


# --------------------------------------------------
# SECTION 4: ROOT CAUSE ANALYSIS
# 1 CHART
# --------------------------------------------------

st.markdown("---")
st.header("4. Root Cause Analysis")

st.write(
    "Explore the top factors associated with delayed deliveries "
    "in a selected region."
)

region_options = sorted(
    filtered_df["Order Region"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_region = st.selectbox(
    "Select Region for Root Cause Analysis",
    region_options
)

def top_drivers_for_region(data, region):

    df_region = data[
        data["Order Region"].astype(str) == region
    ].copy()

    drivers = [
        "Shipping Mode",
        "Customer Segment",
        "Department Name",
        "Type",
        "Order Status"
    ]

    all_factors = []

    for factor in drivers:

        temp = (
            df_region.groupby(factor)
            .agg(
                total_orders=("Delay", "count"),
                late_orders=("Is_Delayed", "sum"),
                avg_delay=("Delay", "mean")
            )
            .reset_index()
        )

        temp["delay_pct"] = (
            temp["late_orders"] /
            temp["total_orders"] * 100
        )

        temp["Driver"] = factor

        temp["Factor_Level"] = (
            factor + " : " + temp[factor].astype(str)
        )

        all_factors.append(
            temp[
                [
                    "Driver",
                    "Factor_Level",
                    "delay_pct",
                    "avg_delay",
                    "total_orders"
                ]
            ]
        )

    final_df = pd.concat(
        all_factors,
        ignore_index=True
    )

    return final_df.sort_values(
        "delay_pct", ascending=False
    ).head(10)


st.subheader(
    f"4.1 Top Drivers of Late Delivery in {selected_region}"
)

top_factors = top_drivers_for_region(
    filtered_df,
    selected_region
)

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.barh(
    top_factors["Factor_Level"],
    top_factors["delay_pct"],
    color=sns.color_palette(
        "viridis",
        len(top_factors)
    )
)

ax.invert_yaxis()

ax.set_xlabel("Delay Percentage (%)")
ax.set_ylabel("Driver Factors")
ax.set_title(
    f"Top Drivers of Late Delivery in {selected_region}"
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        fontsize=9
    )

if not top_factors.empty:
    ax.set_xlim(
        0,
        top_factors["delay_pct"].max() * 1.18
    )

show_chart(fig)

st.caption(
    "These are associations in the selected data, not proof of causation."
)


# --------------------------------------------------
# SECTION 5: TIME-BASED ANALYSIS
# 3 CHARTS
# --------------------------------------------------

st.markdown("---")
st.header("5. Time-Based Analysis")

st.write(
    "Understand how delivery delays vary across months, "
    "days of the week and hours of the day."
)

# Chart 13: Delay % by Month

st.subheader("5.1 Delay % Trend Over Month")

if "order_month" in filtered_df.columns:

    delay_by_month = (
        filtered_df.groupby("order_month")["Is_Delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delay_pct")
    )

    delay_by_month = delay_by_month.sort_values(
        "order_month"
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        delay_by_month["order_month"],
        delay_by_month["delay_pct"],
        marker="o",
        color=primary_color,
        linewidth=2
    )

    ax.set_xticks(range(1, 13))

    ax.set_xticklabels([
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ])

    ax.set_xlabel("Month")
    ax.set_ylabel("Delay Percentage (%)")
    ax.set_title("Delay % Trend Over Month")
    ax.grid(True, linestyle=":", alpha=0.5)

    for _, row in delay_by_month.nlargest(
        3, "delay_pct"
    ).iterrows():

        ax.annotate(
            f"{row['delay_pct']:.1f}%",
            (row["order_month"], row["delay_pct"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=danger_color
        )

    show_chart(fig)

else:
    st.warning("order_month column not found.")


# Chart 14: Delay % by Day of Week

st.subheader("5.2 Delay % by Day of Week")

if "order_day" in filtered_df.columns:

    delay_by_day = (
        filtered_df.groupby("order_day")["Is_Delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delay_pct")
    )

    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    # Keep weekday order if values are day names
    if set(delay_by_day["order_day"].dropna()).issubset(
        set(day_order)
    ):

        delay_by_day["order_day"] = pd.Categorical(
            delay_by_day["order_day"],
            categories=day_order,
            ordered=True
        )

        delay_by_day = delay_by_day.sort_values("order_day")

    fig, ax = plt.subplots(figsize=(10, 4))

    bars = ax.bar(
        delay_by_day["order_day"].astype(str),
        delay_by_day["delay_pct"],
        color=primary_color
    )

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Delay Percentage (%)")
    ax.set_title("Delay % by Day of Week")

    plt.xticks(rotation=30)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            fontsize=9
        )

    show_chart(fig)

else:
    st.warning("order_day column not found.")


# Chart 15: Delay % by Hour

st.subheader("5.3 Delay % by Hour")

if "order_hour" in filtered_df.columns:

    delay_by_hour = (
        filtered_df.groupby("order_hour")["Is_Delayed"]
        .mean()
        .mul(100)
        .reset_index(name="delay_pct")
    )

    delay_by_hour = delay_by_hour.sort_values(
        "order_hour"
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        delay_by_hour["order_hour"],
        delay_by_hour["delay_pct"],
        marker="o",
        color=primary_color,
        linewidth=2
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Delay Percentage (%)")
    ax.set_title("Delay % by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle=":", alpha=0.5)

    for _, row in delay_by_hour.nlargest(
        3, "delay_pct"
    ).iterrows():

        ax.annotate(
            f"{row['delay_pct']:.1f}%",
            (row["order_hour"], row["delay_pct"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=danger_color
        )

    show_chart(fig)

else:
    st.warning("order_hour column not found.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Supply Chain Analytics | Built with Python, Pandas, "
    "Matplotlib, Seaborn and Streamlit"
)
