
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Late Delivery Risk Assessment")

st.markdown(
    """
    Assess order-level late delivery risk using a trained Machine Learning
    model and identify suitable supply chain actions before shipment.
    """
)

st.divider()


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("late_delivery_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# --------------------------------------------------
# RISK BAND CONFIGURATION
# --------------------------------------------------

# These are operational thresholds for displaying risk bands.
# They are not thresholds learned automatically by the ML model.

LOW_RISK_LIMIT = 0.40
HIGH_RISK_LIMIT = 0.70


# --------------------------------------------------
# ORDER INFORMATION FORM
# --------------------------------------------------

st.subheader("📦 Order Information")

st.caption(
    "Enter order details available at the time of order placement. "
    "For categorical fields, use values and spellings consistent with the training data."
)

with st.form("risk_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Order & Product Details**")

        order_type = st.selectbox(
            "Order Type",
            ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
        )

        scheduled_days = st.number_input(
            "Scheduled Shipping Days",
            min_value=0,
            max_value=10,
            value=4
        )

        category = st.text_input(
            "Category Name",
            "Sporting Goods"
        )

        department = st.text_input(
            "Department Name",
            "Fan Shop"
        )

        customer_segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"]
        )

    with col2:
        st.markdown("**Logistics & Order Timing**")

        region = st.text_input(
            "Order Region",
            "Western Europe"
        )

        shipping_mode = st.selectbox(
            "Shipping Mode",
            [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day"
            ]
        )

        order_month = st.slider(
            "Order Month",
            1, 12, 6
        )

        order_day = st.slider(
            "Day of Month",
            1, 31, 15
        )

        order_hour = st.slider(
            "Order Hour",
            0, 23, 12
        )

    submitted = st.form_submit_button(
        "🔍 Assess Delivery Risk",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    # Keep feature names and order consistent with model training
    input_data = pd.DataFrame([{
        "Type": order_type,
        "Days for shipment (scheduled)": scheduled_days,
        "Category Name": category.strip(),
        "Customer Segment": customer_segment,
        "Department Name": department.strip(),
        "Order Region": region.strip(),
        "Shipping Mode": shipping_mode,
        "order_month": order_month,
        "order_day": order_day,
        "order_hour": order_hour
    }])

    if not category.strip() or not department.strip() or not region.strip():
        st.warning("Please enter all categorical order details.")
        st.stop()

    try:
        prediction = model.predict(input_data)[0]

        # Extract probability corresponding to class 1
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            if hasattr(model, "classes_"):
                class_index = list(model.classes_).index(1)
            else:
                class_index = 1

            probability = probabilities[class_index]

        else:
            probability = None

    except Exception as e:
        st.error(
            f"Prediction failed. Please verify that categorical values "
            f"match the model's training data.\n\nDetails: {e}"
        )
        st.stop()

    st.divider()

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------

    st.subheader("📊 Risk Assessment Results")

    if probability is not None:

        if probability < LOW_RISK_LIMIT:
            risk_band = "Low"
            st.success("LOW RISK — Order can follow the standard fulfilment process.")

            recommendation = [
                "Proceed with the planned shipping schedule.",
                "Follow standard shipment tracking procedures.",
                "Maintain regular delivery performance monitoring."
            ]

        elif probability < HIGH_RISK_LIMIT:
            risk_band = "Medium"
            st.warning("MEDIUM RISK — Order requires closer monitoring.")

            recommendation = [
                "Monitor shipment progress at key logistics milestones.",
                "Review the planned dispatch and delivery schedule.",
                "Keep the customer informed if a delay indicator emerges."
            ]

        else:
            risk_band = "High"
            st.error("HIGH RISK — Prioritise this order for operational review.")

            recommendation = [
                "Flag the order for priority review by the logistics team.",
                "Evaluate whether a faster feasible shipping option is available.",
                "Coordinate with the fulfilment and transportation teams.",
                "Proactively communicate potential delivery issues to the customer."
            ]

        # Two-column result display
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Late Delivery Risk Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                "Operational Risk Band",
                risk_band
            )

        st.progress(float(probability))

        st.caption(
            "The probability represents the model's estimated likelihood "
            "of the Late_delivery_risk = 1 class."
        )

    else:
        risk_band = "Not available"

        if prediction == 1:
            st.error("⚠️ Model Prediction: Late Delivery Risk")
        else:
            st.success("✅ Model Prediction: No Late Delivery Risk")

        recommendation = [
            "Review the order's logistics plan.",
            "Monitor shipment progress using standard operational procedures."
        ]

        st.info(
            "This model does not provide probability estimates. "
            "Only its predicted class is displayed."
        )

    # --------------------------------------------------
    # MODEL CLASSIFICATION
    # --------------------------------------------------

    st.markdown("### Model Classification")

    if prediction == 1:
        st.write("**Predicted class:** Late Delivery Risk (1)")
    else:
        st.write("**Predicted class:** No Late Delivery Risk (0)")

    # --------------------------------------------------
    # SCM ACTION RECOMMENDATIONS
    # --------------------------------------------------

    st.divider()

    st.subheader("💡 Recommended Supply Chain Actions")

    st.write(
        "Based on the assessed risk, the following operational actions "
        "can help the supply chain team manage the order:"
    )

    for i, action in enumerate(recommendation, start=1):
        st.markdown(f"{i}. {action}")

    # --------------------------------------------------
    # ORDER SUMMARY
    # --------------------------------------------------

    with st.expander("View Submitted Order Details"):

        st.dataframe(
            input_data.T.rename(columns={0: "Order Information"}),
            use_container_width=True
        )


# --------------------------------------------------
# BUSINESS INTERPRETATION
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ How to Interpret This Assessment"):

    st.markdown(
        """
        **Risk probability bands used in this application:**

        - **Low Risk:** Probability below 40%
        - **Medium Risk:** Probability from 40% to below 70%
        - **High Risk:** Probability of 70% or above

        These are configurable operational thresholds, not universal
        supply chain standards. They should ideally be calibrated using
        historical outcomes, business service-level targets and the
        cost of late deliveries.

        **Important distinction:**

        The model predicts the `Late_delivery_risk` label. It does not
        guarantee whether an order will actually arrive late.

        The recommendations are operational suggestions based on the
        predicted risk band, not proof of the cause of a potential delay.
        """
    )

st.caption(
    "Supply Chain Analytics | Machine Learning-Based Risk Assessment"
)
