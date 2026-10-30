
import streamlit as st

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide"
)

# Header
st.title("Supply Chain Analytics & Late Delivery Prediction")

st.markdown(
    """
    An interactive analytics platform designed to evaluate
    supply chain performance, identify delivery bottlenecks,
    and predict late delivery risk using Machine Learning.
    """
)

st.markdown("---")

# Project Overview
st.header("Project Overview")

st.write(
    """
    This application uses historical supply chain data to analyse
    delivery performance and profitability, identify factors
    associated with delayed deliveries, and provide insights
    to support data-driven operational decisions.
    """
)

# Key Features
st.header("Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Analytics Dashboard")
    st.write(
        "Explore delivery trends, profitability, bottlenecks, "
        "and factors associated with delivery delays."
    )

with col2:
    st.subheader("🤖 Late Delivery Prediction")
    st.write(
        "Predict late delivery risk using a trained "
        "Machine Learning classification model."
    )

with col3:
    st.subheader("💡 Business Recommendations")
    st.write(
        "Explore actionable recommendations to improve "
        "delivery performance and supply chain operations."
    )

st.markdown("---")

# Navigation
st.info(
    "Use the pages in the sidebar to access the Analytics Dashboard, "
    "Late Delivery Prediction, and Business Recommendations."
)

# Footer
st.markdown("---")

st.caption(
    "Supply Chain Analytics | Machine Learning | Data-Driven Decision Making"
)
