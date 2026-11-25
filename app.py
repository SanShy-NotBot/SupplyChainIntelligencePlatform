import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


#1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Supply Chain ML & Operations Dashboard",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Supply Chain Analytics & Predictive Operations Dashboard")
st.markdown(
    "Interactive decision-support platform combining **Supervised Risk/Duration Models** "
    "and **Unsupervised K-Means Operational Segmentation**."
)


# 2. DATA & MODEL LOADING (CACHED)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    clustered_path = os.path.join(BASE_DIR, "clustered_supply_chain.csv")
    cleaned_path = os.path.join(BASE_DIR, "cleaned_supply_chain.csv")
    if os.path.exists(clustered_path):
        return pd.read_csv(clustered_path), True
    return pd.read_csv(cleaned_path), False


@st.cache_resource
def load_artifacts():
    artifacts = {}
    files = {
        "cluster_scaler": "cluster_scaler.joblib",
        "pca_2d": "pca_2d.joblib",
        "kmeans_model": "kmeans_model.joblib",
        "cluster_features": "cluster_features.joblib",
        "cluster_labels": "cluster_labels.joblib",

        # Expected filenames from Shivanshi...
        "reg_model": "random_forest_grouped_regression_model.joblib",
        "clf_model": "best_classification_model.joblib",
    }
    for key, fname in files.items():
        fpath = os.path.join(BASE_DIR, fname)
        artifacts[key] = joblib.load(fpath) if os.path.exists(fpath) else None
    return artifacts


df, has_clusters = load_data()
models = load_artifacts()

def build_supervised_input_df(
    scheduled_days: int,
    shipping_mode: str,
    payment_type: str,
    market: str,
    order_region: str,
    customer_segment: str,
    department_name: str,
    product_price: float,
    quantity: int,
    discount_rate: float,
    expected_profit: float,
    order_month: int,
    order_dayofweek: int,
    order_hour: int,
    is_weekend: int,
    reference_model,
) -> pd.DataFrame:
    """Constructs the exact feature DataFrame (61 cols for Reg, 59 cols for Clf)."""
    sales_val = product_price * quantity
    discount_val = sales_val * discount_rate
    item_total_val = sales_val - discount_val
    profit_ratio_val = (expected_profit / sales_val) if sales_val > 0 else 0.0

    region_subset = df[df["Order Region"] == order_region]
    lat_val = float(region_subset["Latitude"].median()) if not region_subset.empty else float(df["Latitude"].median())
    lon_val = float(region_subset["Longitude"].median()) if not region_subset.empty else float(df["Longitude"].median())

    row = {
        "Days for shipment (scheduled)": scheduled_days,
        "Latitude": lat_val,
        "Longitude": lon_val,
        "Order Item Discount": discount_val,
        "Order Item Discount Rate": discount_rate,
        "Order Item Product Price": product_price,
        "Order Item Profit Ratio": profit_ratio_val,
        "Order Item Quantity": quantity,
        "Sales": sales_val,
        "Order Item Total": item_total_val,
        "Order Profit Per Order": expected_profit,
        "order_year": 2017,
        "order_month": order_month,
        "order_day": 15,
        "order_dayofweek": order_dayofweek,
        "order_hour": order_hour,
        "is_weekend": is_weekend,
        f"Type_{payment_type}": 1,
        f"Shipping Mode_{shipping_mode}": 1,
        f"Market_{market}": 1,
        f"Order Region_{order_region}": 1,
        f"Customer Segment_{customer_segment}": 1,
        f"Department Name_{department_name}": 1,
    }

    expected_cols = list(reference_model.feature_names_in_)
    return pd.DataFrame([row]).reindex(columns=expected_cols, fill_value=0)

# ---------------------------------------------------------
# 3. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Global Operational Filters")

markets = ["All"] + sorted(df["Market"].dropna().unique().tolist())
selected_market = st.sidebar.selectbox("Select Market", markets)

ship_modes = ["All"] + sorted(df["Shipping Mode"].dropna().unique().tolist())
selected_mode = st.sidebar.selectbox("Select Shipping Mode", ship_modes)

filtered_df = df.copy()
if selected_market != "All":
    filtered_df = filtered_df[filtered_df["Market"] == selected_market]
if selected_mode != "All":
    filtered_df = filtered_df[filtered_df["Shipping Mode"] == selected_mode]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Orders", f"{len(filtered_df):,}")
st.sidebar.metric(
    "Avg Late Delivery Rate",
    f"{filtered_df['Late_delivery_risk'].mean() * 100:.1f}%",
)

# ---------------------------------------------------------
# 4. TOP KPI BANNER
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Orders", f"{len(filtered_df):,}")
kpi2.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
kpi3.metric(
    "Avg Actual Shipping Days",
    f"{filtered_df['Days for shipping (real)'].mean():.2f} days",
)
kpi4.metric(
    "Late Delivery Rate",
    f"{filtered_df['Late_delivery_risk'].mean() * 100:.1f}%",
)

st.markdown("---")

# ---------------------------------------------------------
# 5. MAIN DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📊 EDA & Operational Stats",
        "⏱️ Live Delay & Duration Predictor",
        "🧩 Unsupervised Cluster Segments",
    ]
)

# =========================================================
# TAB 1: EDA & OPERATIONAL STATS
# =========================================================
with tab1:
    st.subheader("Supply Chain Performance & Bottleneck Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = (
            filtered_df["Delivery Status"].value_counts().reset_index()
        )
        status_counts.columns = ["Delivery Status", "Count"]
        fig_status = px.pie(
            status_counts,
            names="Delivery Status",
            values="Count",
            hole=0.45,
            title="Order Distribution by Delivery Status",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_b:
        mode_delay = (
            filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )
        mode_delay["Late Risk (%)"] = mode_delay["Late_delivery_risk"] * 100
        fig_mode = px.bar(
            mode_delay,
            x="Shipping Mode",
            y="Late Risk (%)",
            color="Shipping Mode",
            title="Late Delivery Risk (%) by Shipping Mode",
            text_auto=".1f",
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        region_profit = (
            filtered_df.groupby("Order Region")["Order Profit Per Order"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_region = px.bar(
            region_profit,
            x="Order Profit Per Order",
            y="Order Region",
            orientation="h",
            title="Top 10 Regions by Total Profit ($)",
            color="Order Profit Per Order",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col_d:
        monthly_trend = (
            filtered_df.groupby("order_month")["Sales"]
            .sum()
            .reset_index()
        )
        fig_month = px.line(
            monthly_trend,
            x="order_month",
            y="Sales",
            markers=True,
            title="Monthly Sales Volume Trend",
            labels={"order_month": "Order Month", "Sales": "Total Sales ($)"},
        )
        st.plotly_chart(fig_month, use_container_width=True)

# =========================================================
# TAB 2: LIVE PREDICTION ENGINE (SUPERVISED + CLUSTER)
# =========================================================
with tab2:
    st.subheader("Simulate Order Parameters for Live Predictions")
    st.caption(
        "Adjust order characteristics below to predict operational segment, "
        "estimated shipping duration, and late delivery risk."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            scheduled_days = st.slider("Scheduled Shipping Days", 0, 4, 4)
            shipping_mode_in = st.selectbox(
                "Shipping Mode",
                sorted(df["Shipping Mode"].dropna().unique().tolist()),
            )
            payment_type_in = st.selectbox(
                "Payment Type",
                sorted(df["Type"].dropna().unique().tolist()),
            )
            market_in = st.selectbox(
                "Market",
                sorted(df["Market"].dropna().unique().tolist()),
            )
            order_region_in = st.selectbox(
                "Order Region",
                sorted(df["Order Region"].dropna().unique().tolist()),
            )

        with c2:
            customer_segment_in = st.selectbox(
                "Customer Segment",
                sorted(df["Customer Segment"].dropna().unique().tolist()),
            )
            department_in = st.selectbox(
                "Department Name",
                sorted(df["Department Name"].dropna().unique().tolist()),
            )
            product_price = st.number_input("Product Price ($)", 10.0, 2000.0, 150.0)
            quantity = st.slider("Order Item Quantity", 1, 5, 2)
            discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.10, step=0.01)

        with c3:
            order_month_in = st.slider("Order Month (1-12)", 1, 12, 6)
            order_dayofweek_in = st.selectbox(
                "Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
            )
            order_hour_in = st.slider("Order Hour of Day (0-23)", 0, 23, 12)
            expected_profit = st.number_input("Expected Profit ($)", -400.0, 900.0, 25.0)
            est_real_days = st.slider(
                "Current/Estimated Transit Days (for Segment Profiling)", 0, 6, 3
            )

        submitted = st.form_submit_button("🚀 Run Live Prediction")

    if submitted:
        sales_val = product_price * quantity
        prelim_delay_flag = 1 if est_real_days > scheduled_days else 0
        is_weekend_in = 1 if order_dayofweek_in in [5, 6] else 0

        res1, res2, res3 = st.columns(3)

        # 1. Unsupervised Cluster Assignment (Your Model)
        with res1:
            st.markdown("### 🧩 Operational Cluster")
            if all(
                models[k] is not None
                for k in ["cluster_scaler", "pca_2d", "kmeans_model", "cluster_features", "cluster_labels"]
            ):
                input_df = pd.DataFrame(
                    [
                        [
                            est_real_days,
                            scheduled_days,
                            sales_val,
                            quantity,
                            discount_rate,
                            expected_profit,
                            product_price,
                            prelim_delay_flag,
                            is_weekend_in,
                            order_hour_in,
                        ]
                    ],
                    columns=models["cluster_features"],
                )
                scaled_vec = models["cluster_scaler"].transform(input_df)
                pca_vec = models["pca_2d"].transform(scaled_vec)
                pred_cluster = int(models["kmeans_model"].predict(pca_vec)[0])
                seg_name = models["cluster_labels"].get(
                    pred_cluster, f"Cluster {pred_cluster}"
                )
                st.success(f"**Cluster {pred_cluster}: {seg_name}**")
            else:
                st.warning(
                    "Run `code4_clustering.ipynb` first to generate `kmeans_model.joblib`."
                )

        is_weekend_in = 1 if order_dayofweek_in in [5, 6] else 0

        # 2. Teammate A: Regression Duration Prediction
        with res2:
            st.markdown("### ⏱️ Predicted Shipping Days")
            if models["reg_model"] is not None:
                sup_input_reg = build_supervised_input_df(
                    scheduled_days=scheduled_days,
                    shipping_mode=shipping_mode_in,
                    payment_type=payment_type_in,
                    market=market_in,
                    order_region=order_region_in,
                    customer_segment=customer_segment_in,
                    department_name=department_in,
                    product_price=product_price,
                    quantity=quantity,
                    discount_rate=discount_rate,
                    expected_profit=expected_profit,
                    order_month=order_month_in,
                    order_dayofweek=order_dayofweek_in,
                    order_hour=order_hour_in,
                    is_weekend=is_weekend_in,
                    reference_model=models["reg_model"],
                )
                pred_days = float(models["reg_model"].predict(sup_input_reg)[0])
                delta_days = pred_days - scheduled_days
                st.metric(
                    "Estimated Actual Transit",
                    f"{pred_days:.2f} days",
                    delta=f"{delta_days:+.2f} days vs SLA",
                    delta_color="inverse",
                )
                st.caption("Random Forest Regressor (Test MAE: 0.99 days | R²: 0.38)")
            else:
                st.warning("`best_regression_model.joblib` not found in project folder.")

        # 3. Teammate A: Classification Late Risk Prediction
        with res3:
            st.markdown("### ⚠️ Late Delivery Risk")
            if models["clf_model"] is not None:
                sup_input_clf = build_supervised_input_df(
                    scheduled_days=scheduled_days,
                    shipping_mode=shipping_mode_in,
                    payment_type=payment_type_in,
                    market=market_in,
                    order_region=order_region_in,
                    customer_segment=customer_segment_in,
                    department_name=department_in,
                    product_price=product_price,
                    quantity=quantity,
                    discount_rate=discount_rate,
                    expected_profit=expected_profit,
                    order_month=order_month_in,
                    order_dayofweek=order_dayofweek_in,
                    order_hour=order_hour_in,
                    is_weekend=is_weekend_in,
                    reference_model=models["clf_model"],
                )
                pred_class = int(models["clf_model"].predict(sup_input_clf)[0])
                late_prob = float(models["clf_model"].predict_proba(sup_input_clf)[0, 1])

                if pred_class == 1:
                    st.error(f"**HIGH RISK OF DELAY ({late_prob * 100:.1f}%)**")
                else:
                    st.success(f"**ON-TIME LIKELY (Delay Risk: {late_prob * 100:.1f}%)**")
                st.caption("Tuned RF Classifier (Test Acc: 71.61% | F1: 71.24%)")
            else:
                st.warning("`best_classification_model.joblib` not found in project folder.")

    st.markdown("---")
    with st.expander("📈 Supervised Model Evaluation Benchmarks (Grouped Split by Order ID)"):
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Tuned Random Forest Classifier (`Late_delivery_risk`)**")
            st.write("- **Accuracy:** 71.61% | **Weighted F1:** 71.24%")
            st.write("- **Weighted Precision:** 75.37% | **Weighted Recall:** 71.61%")
            cm_df = pd.DataFrame(
                [[14205, 1995], [8187, 11482]],
                index=["Actual On-Time (0)", "Actual Late (1)"],
                columns=["Pred On-Time (0)", "Pred Late (1)"],
            )
            st.dataframe(cm_df, use_container_width=True)
        with b2:
            st.markdown("**Random Forest Regressor (`Days for shipping (real)`)**")
            st.write("- **MAE:** 0.9899 days")
            st.write("- **RMSE:** 1.2875 days")
            st.write("- **R² Score:** 0.3809")

# =========================================================
# TAB 3: UNSUPERVISED CLUSTER SEGMENTS
# =========================================================
with tab3:
    st.subheader("PCA + K-Means Operational Segmentation")

    if not has_clusters or "Cluster" not in df.columns:
        st.error(
            "`clustered_supply_chain.csv` not found. Please run `code4_clustering.ipynb` first."
        )
    else:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            plot_sample = filtered_df.sample(
                n=min(5000, len(filtered_df)), random_state=42
            )
            fig_pca = px.scatter(
                plot_sample,
                x="PCA1",
                y="PCA2",
                color="Cluster_Name",
                title="2D PCA Projection of Shipment Clusters (Sampled 5,000 Orders)",
                opacity=0.65,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_right:
            seg_counts = (
                filtered_df["Cluster_Name"].value_counts().reset_index()
            )
            seg_counts.columns = ["Segment", "Orders"]
            fig_seg_bar = px.bar(
                seg_counts,
                x="Orders",
                y="Segment",
                orientation="h",
                color="Segment",
                title="Orders per Cluster Segment",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_seg_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_seg_bar, use_container_width=True)

        st.markdown("### 📋 Cluster Trait Summary Table")
        summary_cols = [
            "Days for shipping (real)",
            "Days for shipment (scheduled)",
            "Sales",
            "Order Item Quantity",
            "Order Profit Per Order",
            "Order Item Product Price",
            "Late_delivery_risk",
        ]
        summary_table = (
            filtered_df.groupby(["Cluster", "Cluster_Name"])[summary_cols]
            .mean()
            .round(2)
            .reset_index()
        )
        st.dataframe(summary_table, use_container_width=True)