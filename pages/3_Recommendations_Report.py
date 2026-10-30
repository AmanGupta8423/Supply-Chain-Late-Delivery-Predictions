
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="SCM Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Supply Chain Improvement Recommendations")

st.markdown(
    """
    A decision-support page translating supply chain performance
    metrics into actionable operational improvements.

    Recommendations focus on delivery reliability, transportation
    efficiency, order fulfilment and profitability.
    """
)

st.markdown("---")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "DataCoSupplyChainDataset2.csv",
        encoding="latin1"
    )

    df.columns = df.columns.str.strip()

    df["Days for shipment (scheduled)"] = pd.to_numeric(
        df["Days for shipment (scheduled)"],
        errors="coerce"
    )

    df["Days for shipping (real)"] = pd.to_numeric(
        df["Days for shipping (real)"],
        errors="coerce"
    )

    df["Order Profit Per Order"] = pd.to_numeric(
        df["Order Profit Per Order"],
        errors="coerce"
    )

    df["Delay"] = (
        df["Days for shipping (real)"]
        - df["Days for shipment (scheduled)"]
    )

    df["Is_Delayed"] = (df["Delay"] > 0).astype(int)

    return df


df = load_data()


# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.header("Analysis Scope")

filter_columns = [
    "Order Region",
    "Customer Segment",
    "Shipping Mode",
    "Department Name"
]

filter_cols = st.columns(4)

selected_filters = {}

for i, col in enumerate(filter_columns):

    options = sorted(
        df[col].dropna().astype(str).unique().tolist()
    )

    with filter_cols[i]:

        selected_filters[col] = st.selectbox(
            col,
            ["All"] + options
        )

filtered_df = df.copy()

for col, value in selected_filters.items():

    if value != "All":

        filtered_df = filtered_df[
            filtered_df[col].astype(str) == value
        ]

if filtered_df.empty:

    st.warning("No records available for the selected filters.")
    st.stop()


# --------------------------------------------------
# PERFORMANCE INDICATORS
# --------------------------------------------------

total_orders = len(filtered_df)

delayed_orders = filtered_df["Is_Delayed"].sum()

delay_rate = (
    delayed_orders / total_orders * 100
    if total_orders > 0 else 0
)

avg_delay = filtered_df.loc[
    filtered_df["Is_Delayed"] == 1,
    "Delay"
].mean()

avg_scheduled_days = filtered_df[
    "Days for shipment (scheduled)"
].mean()

avg_actual_days = filtered_df[
    "Days for shipping (real)"
].mean()

total_profit = filtered_df[
    "Order Profit Per Order"
].sum()

avg_profit = filtered_df[
    "Order Profit Per Order"
].mean()


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.markdown("---")
st.header("Supply Chain Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Orders",
    f"{total_orders:,}"
)

c2.metric(
    "Delayed Orders",
    f"{delayed_orders:,}"
)

c3.metric(
    "Delay Rate",
    f"{delay_rate:.1f}%"
)

c4.metric(
    "Average Delay",
    f"{avg_delay:.2f} days"
    if pd.notna(avg_delay) else "N/A"
)

c5, c6, c7 = st.columns(3)

c5.metric(
    "Avg. Scheduled Shipping",
    f"{avg_scheduled_days:.2f} days"
)

c6.metric(
    "Avg. Actual Shipping",
    f"{avg_actual_days:.2f} days"
)

c7.metric(
    "Total Order Profit",
    f"${total_profit:,.0f}"
)

st.caption(
    f"Recommendations are based on {total_orders:,} filtered orders."
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_delay_rate(data, category):

    result = (
        data.groupby(category)
        .agg(
            total_orders=("Is_Delayed", "count"),
            delayed_orders=("Is_Delayed", "sum")
        )
        .reset_index()
    )

    result["delay_rate"] = (
        result["delayed_orders"]
        / result["total_orders"] * 100
    )

    return result


recommendations = []


def add_recommendation(
    priority,
    area,
    observation,
    recommendation,
    expected_impact
):

    recommendations.append({
        "Priority": priority,
        "Area": area,
        "Observation": observation,
        "Recommended Action": recommendation,
        "Expected Impact": expected_impact
    })


# --------------------------------------------------
# 1. DELIVERY PERFORMANCE
# --------------------------------------------------

st.markdown("---")
st.header("1. Delivery Performance Improvement")

if delay_rate > 30:

    add_recommendation(
        "High",
        "Delivery Reliability",
        f"{delay_rate:.1f}% of orders are delayed.",
        "Introduce delivery control-tower monitoring, "
        "exception alerts and daily review of delayed shipments. "
        "Prioritise orders approaching their committed delivery date.",
        "Improved delivery reliability and fewer delayed orders."
    )

elif delay_rate > 15:

    add_recommendation(
        "Medium",
        "Delivery Reliability",
        f"Delay rate is {delay_rate:.1f}%.",
        "Review recurring delay patterns by region and shipping mode. "
        "Introduce weekly carrier performance reviews and "
        "proactive shipment tracking.",
        "Better delivery visibility and operational control."
    )

else:

    add_recommendation(
        "Low",
        "Delivery Reliability",
        f"Delay rate is {delay_rate:.1f}%.",
        "Maintain existing delivery controls and monitor "
        "service-level performance to prevent deterioration.",
        "Sustained delivery performance."
    )


# --------------------------------------------------
# 2. TRANSPORTATION AND SHIPPING
# --------------------------------------------------

st.markdown("---")
st.header("2. Transportation & Shipping Optimization")

shipping_analysis = get_delay_rate(
    filtered_df,
    "Shipping Mode"
)

if not shipping_analysis.empty:

    worst_shipping = shipping_analysis.sort_values(
        "delay_rate",
        ascending=False
    ).iloc[0]

    st.write("**Shipping Mode Performance**")

    st.dataframe(
        shipping_analysis.sort_values(
            "delay_rate",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    if worst_shipping["delay_rate"] > delay_rate:

        add_recommendation(
            "High",
            "Transportation",
            f"{worst_shipping['Shipping Mode']} has a "
            f"{worst_shipping['delay_rate']:.1f}% delay rate.",
            "Conduct a carrier and lane-level performance review "
            "for this shipping mode. Evaluate transit-time variability, "
            "handover delays and route reliability before reallocating "
            "shipment volumes.",
            "Improved carrier accountability and more reliable transit times."
        )


# --------------------------------------------------
# 3. REGIONAL BOTTLENECKS
# --------------------------------------------------

st.markdown("---")
st.header("3. Regional Bottleneck Management")

region_analysis = get_delay_rate(
    filtered_df,
    "Order Region"
)

if not region_analysis.empty:

    worst_region = region_analysis.sort_values(
        "delay_rate",
        ascending=False
    ).iloc[0]

    st.write("**Regional Delivery Performance**")

    st.dataframe(
        region_analysis.sort_values(
            "delay_rate",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    if worst_region["delay_rate"] > delay_rate:

        add_recommendation(
            "High",
            "Regional Operations",
            f"{worst_region['Order Region']} records a "
            f"{worst_region['delay_rate']:.1f}% delay rate.",
            "Investigate regional distribution constraints, "
            "transportation connectivity, fulfilment capacity "
            "and last-mile delivery performance. Consider "
            "regional safety stock or alternate fulfilment locations "
            "only after validating demand and service requirements.",
            "Reduced regional delivery bottlenecks."
        )


# --------------------------------------------------
# 4. CUSTOMER SEGMENT
# --------------------------------------------------

st.markdown("---")
st.header("4. Customer Service Level Management")

segment_analysis = get_delay_rate(
    filtered_df,
    "Customer Segment"
)

if not segment_analysis.empty:

    worst_segment = segment_analysis.sort_values(
        "delay_rate",
        ascending=False
    ).iloc[0]

    st.write("**Customer Segment Performance**")

    st.dataframe(
        segment_analysis.sort_values(
            "delay_rate",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    if worst_segment["delay_rate"] > delay_rate:

        add_recommendation(
            "Medium",
            "Customer Service",
            f"{worst_segment['Customer Segment']} has a "
            f"{worst_segment['delay_rate']:.1f}% delay rate.",
            "Review service-level commitments, order handling "
            "and fulfilment priorities for this customer segment. "
            "Align delivery promises with actual logistics capability.",
            "Improved customer experience and service-level adherence."
        )


# --------------------------------------------------
# 5. PROFITABILITY
# --------------------------------------------------

st.markdown("---")
st.header("5. Profitability & Operational Efficiency")

if total_profit < 0:

    add_recommendation(
        "High",
        "Profitability",
        f"Aggregate order profit is ${total_profit:,.0f}.",
        "Perform order-level profitability analysis covering "
        "transportation cost, discounts, fulfilment expenses "
        "and product margins. Identify loss-making order segments "
        "before changing pricing or service policies.",
        "Better cost visibility and protection of order margins."
    )

else:

    add_recommendation(
        "Medium",
        "Profitability",
        f"Aggregate order profit is ${total_profit:,.0f}.",
        "Analyse profitability across shipping modes and delay "
        "categories. Evaluate whether premium transportation "
        "options generate sufficient incremental margin "
        "to justify their cost.",
        "Improved transportation cost-to-service trade-off."
    )


# --------------------------------------------------
# 6. INVENTORY AND FULFILMENT
# --------------------------------------------------

st.markdown("---")
st.header("6. Inventory & Order Fulfilment")

add_recommendation(
    "Medium",
    "Inventory Planning",
    "Delivery delays can be influenced by inventory availability "
    "and fulfilment readiness; this dataset does not directly "
    "confirm stockout events.",
    "Integrate inventory availability, SKU-level demand forecasts "
    "and warehouse fulfilment data. Monitor stockout frequency, "
    "order fulfilment cycle time and backorders to identify "
    "inventory-related delivery constraints.",
    "Improved fulfilment readiness and inventory visibility."
)


# --------------------------------------------------
# 7. ACTIONABLE RECOMMENDATION SUMMARY
# --------------------------------------------------

st.markdown("---")
st.header("7. Prioritized Action Plan")

recommendation_df = pd.DataFrame(recommendations)

priority_order = {
    "High": 0,
    "Medium": 1,
    "Low": 2
}

recommendation_df["_priority_order"] = (
    recommendation_df["Priority"].map(priority_order)
)

recommendation_df = (
    recommendation_df
    .sort_values("_priority_order")
    .drop(columns="_priority_order")
)

for _, row in recommendation_df.iterrows():

    with st.expander(
        f"{row['Priority']} Priority | {row['Area']}",
        expanded=(row["Priority"] == "High")
    ):

        st.markdown("**Performance Observation**")
        st.write(row["Observation"])

        st.markdown("**Recommended SCM Action**")
        st.write(row["Recommended Action"])

        st.markdown("**Expected Operational Impact**")
        st.write(row["Expected Impact"])


st.markdown("---")

st.caption(
    "Note: Recommendations are generated from observed dataset "
    "patterns. Operational actions should be validated using "
    "carrier, inventory, transportation cost and warehouse-level data "
    "before implementation."
)
