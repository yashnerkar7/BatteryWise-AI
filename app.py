import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="BatteryWise AI | Battery Intelligence", page_icon="🔋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#0b1020;color:#edf2f7;}
[data-testid="stSidebar"] {background:#10182b;border-right:1px solid #202c43;}
[data-testid="stSidebar"] * {color:#e8eef8;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1500px;}
.hero {background:linear-gradient(115deg,#17243a,#142d35);border:1px solid #294657;border-radius:18px;padding:26px 30px;margin-bottom:22px;}
.hero h1 {color:#f5f8ff;font-size:34px;margin:0 0 8px;font-weight:800;}
.hero p {color:#aab9cc;margin:0;font-size:15px;}
.metric-card {background:#141e32;border:1px solid #26344d;border-radius:15px;padding:18px 20px;min-height:112px;}
.metric-label {color:#9aabc2;font-size:13px;font-weight:600;margin-bottom:9px;}
.metric-value {color:#f4f7fc;font-size:27px;font-weight:800;}
.metric-note {color:#8294ad;font-size:12px;margin-top:5px;}
.section-heading {color:#f1f5fb;font-size:21px;font-weight:700;margin:24px 0 12px;}
div[data-testid="stForm"] {background:#141e32;border:1px solid #26344d;border-radius:16px;padding:20px;}
.stButton>button,.stFormSubmitButton>button {background:#a3e635;color:#101827;border:0;border-radius:9px;font-weight:800;min-height:43px;}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#bef264;color:#101827;border:0;}
div[data-testid="stMetric"] {background:#141e32;border:1px solid #26344d;padding:15px;border-radius:13px;}
hr {border-color:#26344d;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("battery_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("batterywise_model.pkl")

try:
    data = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.info("Keep battery_data.csv and batterywise_model.pkl in the same folder as app.py.")
    st.stop()

features = [
    "brand", "model_year", "os", "usage_type", "daily_usage_hours",
    "charging_cycles", "avg_charge_limit_percent", "battery_age_months",
    "overheating_issues", "performance_rating"
]
required = features + ["battery_health_percent"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

if "battery_health_category" not in data.columns:
    def classify_health(value):
        if value < 60:
            return "Low"
        if value < 80:
            return "Moderate"
        return "High"
    data["battery_health_category"] = data["battery_health_percent"].apply(classify_health)

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>""", unsafe_allow_html=True)

def make_prediction(values):
    new_row = pd.DataFrame([values])
    combined = pd.concat([data[features], new_row], ignore_index=True)
    encoded = pd.get_dummies(
        combined,
        columns=["brand", "os", "usage_type", "overheating_issues"],
        drop_first=True
    )
    encoded_input = encoded.iloc[[-1]]
    encoded_input = encoded_input.reindex(columns=list(model.feature_names_in_), fill_value=0)
    prediction = model.predict(encoded_input)[0]
    probabilities = model.predict_proba(encoded_input)[0] if hasattr(model, "predict_proba") else None
    return prediction, probabilities

with st.sidebar:
    st.markdown("## 🔋 BatteryWise AI")
    st.caption("BATTERY INTELLIGENCE PLATFORM")
    st.divider()
    page = st.radio("NAVIGATION", ["Dashboard", "Battery Diagnostics", "Analytics", "Battery Care", "About the Model"])
    st.divider()
    st.markdown("**Project status**")
    st.markdown("<span style='color:#a3e635'>●</span> Local prototype", unsafe_allow_html=True)
    st.caption(f"Dataset: {len(data)} sample records")

if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Battery Intelligence Dashboard</h1><p>Explore laptop battery condition patterns and generate an ML-based estimate.</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("DEVICES IN DATASET", str(len(data)), "Available sample records"),
        ("AVERAGE BATTERY HEALTH", f"{data['battery_health_percent'].mean():.1f}%", "Dataset average"),
        ("AVERAGE CHARGING CYCLES", f"{data['charging_cycles'].mean():.0f}", "Dataset average"),
        ("HIGH HEALTH RECORDS", str((data["battery_health_category"] == "High").sum()), "Category: High")
    ]
    for col, item in zip(cols, metrics):
        with col:
            metric_card(*item)

    st.markdown('<div class="section-heading">Battery health overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, .85])
    with left:
        fig = px.histogram(data, x="battery_health_percent", nbins=12, title="Battery health distribution",
                           labels={"battery_health_percent":"Battery health (%)"})
        fig.update_traces(marker_color="#a3e635", marker_line_color="#0b1020")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#141e32", plot_bgcolor="#141e32",
                          font_color="#c4d0df", margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = data["battery_health_category"].value_counts().rename_axis("Category").reset_index(name="Devices")
        fig = px.pie(counts, names="Category", values="Devices", hole=.62, title="Health category breakdown")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#141e32", font_color="#c4d0df",
                          margin=dict(l=15,r=15,t=55,b=15), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-heading">Battery usage patterns</div>', unsafe_allow_html=True)
    fig = px.scatter(data, x="charging_cycles", y="battery_health_percent",
                     color="battery_health_category", hover_data=["brand","model_year","battery_age_months"],
                     title="Charging cycles vs battery health")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#141e32", plot_bgcolor="#141e32",
                      font_color="#c4d0df", margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Statistics describe the sample dataset, not the current battery of the computer running this app.")

elif page == "Battery Diagnostics":
    st.markdown('<div class="hero"><h1>Battery Diagnostics</h1><p>Enter laptop usage and battery details to receive an ML-based category estimate.</p></div>', unsafe_allow_html=True)
    st.info("This tool estimates a category from dataset patterns. It does not directly measure actual battery capacity.")
    with st.form("diagnostic_form"):
        st.markdown("### Device information")
        a, b = st.columns(2)
        with a:
            brand = st.selectbox("Laptop brand", sorted(data["brand"].dropna().unique()))
            model_year = st.number_input("Laptop model year", min_value=int(data.model_year.min()),
                                         max_value=int(data.model_year.max()), value=int(data.model_year.median()))
            os_name = st.selectbox("Operating system", sorted(data["os"].dropna().unique()))
            usage_type = st.selectbox("Primary usage", sorted(data["usage_type"].dropna().unique()))
            daily_usage_hours = st.slider("Daily usage (hours)", 0.0, 24.0, 6.0, .5)
        with b:
            charging_cycles = st.number_input("Charging cycles", min_value=0, value=250, step=10)
            charge_limit = st.slider("Average charge limit (%)", 1, 100, 80)
            battery_age_months = st.number_input("Battery age (months)", min_value=0, value=18)
            overheating = st.selectbox("Overheating issues", sorted(data["overheating_issues"].dropna().unique()))
            performance_rating = st.number_input("Performance rating",
                min_value=int(data.performance_rating.min()), max_value=int(data.performance_rating.max()),
                value=int(data.performance_rating.median()))
        submitted = st.form_submit_button("Run battery assessment", use_container_width=True)

    if submitted:
        values = {
            "brand":brand, "model_year":model_year, "os":os_name, "usage_type":usage_type,
            "daily_usage_hours":daily_usage_hours, "charging_cycles":charging_cycles,
            "avg_charge_limit_percent":charge_limit, "battery_age_months":battery_age_months,
            "overheating_issues":overheating, "performance_rating":performance_rating
        }
        try:
            prediction, probabilities = make_prediction(values)
            st.session_state["last_prediction"] = {"inputs":values,"prediction":prediction,"probabilities":probabilities}
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.markdown("### Assessment result")
        prediction = result["prediction"]
        color = {"High":"#a3e635","Moderate":"#fbbf24","Low":"#fb7185"}.get(prediction,"#c4d0df")
        st.markdown(f"""
        <div style="background:#141e32;border:1px solid #26344d;border-radius:16px;padding:24px;">
          <div style="color:#9aabc2;font-size:13px;font-weight:700;">PREDICTED BATTERY HEALTH CATEGORY</div>
          <div style="font-size:38px;font-weight:800;color:{color};margin:8px 0;">{prediction}</div>
          <div style="color:#c4d0df;">Model-generated category estimate</div>
        </div>""", unsafe_allow_html=True)
        probabilities = result["probabilities"]
        if probabilities is not None:
            probability_df = pd.DataFrame({"Category":model.classes_,"Model probability":probabilities})
            fig = px.bar(probability_df, x="Category", y="Model probability", color="Category",
                         text=probability_df["Model probability"].map(lambda x:f"{x:.1%}"),
                         title="Model output probabilities")
            fig.update_yaxes(tickformat=".0%", range=[0,1])
            fig.update_layout(template="plotly_dark",paper_bgcolor="#141e32",plot_bgcolor="#141e32",
                              font_color="#c4d0df",showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        report = pd.DataFrame([{**result["inputs"],"predicted_category":prediction}])
        st.download_button("Download assessment summary (CSV)", report.to_csv(index=False).encode("utf-8"),
                           "batterywise_assessment.csv","text/csv")
        st.caption("Probabilities are model outputs, not calibrated certainty. This small educational dataset is not validated for real-world diagnosis.")

elif page == "Analytics":
    st.markdown('<div class="hero"><h1>Battery Analytics</h1><p>Explore relationships between battery health, age, usage and charging cycles.</p></div>', unsafe_allow_html=True)
    brands = ["All"] + sorted(data["brand"].dropna().unique().tolist())
    selected_brand = st.selectbox("Filter by laptop brand", brands)
    filtered = data if selected_brand == "All" else data[data.brand == selected_brand]
    m1,m2,m3 = st.columns(3)
    m1.metric("Records shown",len(filtered))
    m2.metric("Average health",f"{filtered.battery_health_percent.mean():.1f}%")
    m3.metric("Average battery age",f"{filtered.battery_age_months.mean():.1f} months")
    x1,x2 = st.columns(2)
    with x1:
        fig = px.scatter(filtered,x="battery_age_months",y="battery_health_percent",
                         color="battery_health_category",hover_data=["brand","charging_cycles"],
                         title="Battery age vs health")
        fig.update_layout(template="plotly_dark",paper_bgcolor="#141e32",plot_bgcolor="#141e32")
        st.plotly_chart(fig,use_container_width=True)
    with x2:
        fig = px.box(filtered,x="usage_type",y="battery_health_percent",color="usage_type",
                     title="Battery health by usage type")
        fig.update_layout(template="plotly_dark",paper_bgcolor="#141e32",plot_bgcolor="#141e32")
        st.plotly_chart(fig,use_container_width=True)
    brand_means = filtered.groupby("brand",as_index=False)["battery_health_percent"].mean()
    fig = px.bar(brand_means,x="brand",y="battery_health_percent",title="Average battery health by brand")
    fig.update_layout(template="plotly_dark",paper_bgcolor="#141e32",plot_bgcolor="#141e32")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("These are descriptive patterns in the sample dataset; they do not prove causation.")

elif page == "Battery Care":
    st.markdown('<div class="hero"><h1>Battery Care Centre</h1><p>Practical habits that can help reduce unnecessary battery wear.</p></div>', unsafe_allow_html=True)
    tips = [
        ("Use sensible charge limits","If your laptop supports a charge limit, consider the manufacturer's recommended battery conservation setting."),
        ("Avoid excessive heat","Keep vents clear and avoid using the laptop on soft surfaces that block airflow."),
        ("Use the correct charger","Use a compatible charger that meets your laptop manufacturer's specifications."),
        ("Avoid prolonged extreme charge levels","For long-term storage, follow your laptop manufacturer's battery storage guidance."),
        ("Check battery reports","Use your operating system's battery tools to review actual capacity and cycle information.")
    ]
    for title,description in tips:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    st.info("Battery settings and recommendations vary by laptop model. Check your manufacturer's documentation.")

elif page == "About the Model":
    st.markdown('<div class="hero"><h1>About BatteryWise AI</h1><p>Understand the data, model workflow and limitations behind this prototype.</p></div>', unsafe_allow_html=True)
    st.markdown("### How the prediction works")
    st.write("The application collects laptop and usage information, converts categorical fields into numerical features and passes the result to a trained Decision Tree Classifier.")
    st.markdown("### Health categories")
    st.write("- **Low:** below 60%")
    st.write("- **Moderate:** 60% to below 80%")
    st.write("- **High:** 80% or above")
    st.markdown("### Model inputs")
    st.write(", ".join(features))
    st.warning("This prototype uses only 50 records, with relatively few Low-category examples. Predictions are educational estimates, not professional diagnostics or actual battery-capacity measurements. The thresholds are project-defined.")
    st.markdown("### Dataset preview")
    st.dataframe(data.head(10),use_container_width=True)
    st.markdown("### Dataset category counts")
    st.dataframe(data["battery_health_category"].value_counts().rename_axis("Category").reset_index(name="Records"),use_container_width=True)
