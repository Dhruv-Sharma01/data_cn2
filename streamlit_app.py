import streamlit as st
import pandas as pd
import json
import joblib
import graphviz
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Center Predictive Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Feature Lists (Copied from ml_pipeline_all_tasks_FIXED.py) ---
# These are CRITICAL for the models to work, as they must match
# the exact features the model was trained on.

FEATURES_CONGESTION = [
    "protocol_cat", "avg_bps", "avg_jitter_ms", "avg_rtt_ms",
    "client_buffer_bytes", "server_buffer_bytes"
]
FEATURES_SCENARIO = [
    "protocol_cat", "avg_bps", "total_packets", "total_lost_or_retrans",
    "packet_loss_rate", "avg_jitter_ms", "avg_rtt_ms", "bdp_bytes"
]
FEATURES_RTT = [
    "protocol_cat", "avg_bps", "total_packets", "total_lost_or_retrans",
    "packet_loss_rate", "avg_jitter_ms",
    "client_buffer_bytes", "server_buffer_bytes"
]
FEATURES_THROUGHPUT = [
    "protocol_cat", "total_packets", "total_lost_or_retrans",
    "packet_loss_rate", "avg_jitter_ms", "avg_rtt_ms",
    "client_buffer_bytes", "server_buffer_bytes"
]
FEATURES_BUFFER = [
    "protocol_cat", "avg_bps", "avg_rtt_ms", "avg_jitter_ms",
    "packet_loss_rate", "client_buffer_bytes", "server_buffer_bytes"
]

# --- Data & Model Loading (Cached for Performance) ---

@st.cache_resource
def load_models():
    """Loads all 5 ML models from the 'models' directory."""
    models = {}
    model_paths = {
        'congestion': 'models_combined/clf_congestion.joblib',
        'scenario': 'models_combined/clf_scenario.joblib',
        'rtt': 'models_combined/reg_rtt.joblib',
        'throughput': 'models_combined/reg_throughput.joblib',
        'buffer': 'models_combined/reg_buffer.joblib'
    }
    for name, path in model_paths.items():
        if Path(path).exists():
            models[name] = joblib.load(path)
        else:
            st.error(f"Error: Model file not found at {path}")
            models[name] = None
    return models

@st.cache_data
def load_data():
    """Loads all CSVs and the topology JSON from the 'data' directory."""
    data = {}
    # Load summary CSVs
    try:
        df_tcp = pd.read_csv("./tcp_test/summary_tcp.csv")
        df_udp = pd.read_csv("./udp_test/summary_udp.csv")
        # Ensure protocol_cat is set for filtering and model input
        df_tcp['protocol_cat'] = 1
        df_udp['protocol_cat'] = 0
        data['main_df'] = pd.concat([df_tcp, df_udp], ignore_index=True)
    except FileNotFoundError as e:
        st.error(f"Error loading summary CSVs from 'data/': {e}")
        data['main_df'] = pd.DataFrame()

    # Load topology
    try:
        with open("./Sonic.json", 'r') as f:
            data['topology'] = json.load(f)
    except FileNotFoundError as e:
        st.error(f"Error: Sonic.json not found at 'data/Sonic.json'")
        data['topology'] = None

    return data

@st.cache_data
def create_topology_graph(topology_data):
    """Creates a Graphviz chart from the Sonic.json topology."""
    if not topology_data or 'content' not in topology_data:
        return None

    g = graphviz.Digraph(engine='dot')
    g.attr(bgcolor='transparent', rankdir='TB')

    nodes = topology_data['content']['nodes']
    links = topology_data['content']['links']

    # Add nodes with styling
    for name, details in nodes.items():
        if 'leaf' in name:
            g.node(name, shape='box', style='filled', fillcolor='#AED6F1') # Blue
        elif 'spine' in name:
            g.node(name, shape='box', style='filled', fillcolor='#ABEBC6') # Green
        elif 'server' in name:
            g.node(name, shape='ellipse', style='filled', fillcolor='#EAECEE') # Gray
        # We can skip oob-mgmt-server etc. for a cleaner look
        elif 'oob' not in name:
            g.node(name, shape='box', style='filled', fillcolor='lightgray')

    # Add edges
    for link_pair in links:
        node1 = link_pair[0]['node']
        node2 = link_pair[1]['node']
        # Add edge only if both nodes are in our main topology (skip mgmt)
        if node1 in nodes and node2 in nodes:
            g.edge(node1, node2)

    return g


# --- Main Application ---
models = load_models()
data = load_data()
df_main = data['main_df']

st.title("Data Center Predictive Health Dashboard")

# --- Define Tabs ---
tab1, tab2 = st.tabs(["NOC Dashboard", "ML Model Insights"])

# --- TAB 1: NOC DASHBOARD ---
with tab1:
    # --- Sidebar Filters ---
    st.sidebar.header("Global Filters")
    protocol = st.sidebar.radio(
        "Protocol",
        ["All", "TCP", "UDP"],
        index=0
    )
    scenario_map = {
        "All": "All",
        "A (Normal)": "A",
        "B (Congested)": "B",
        "C (Failover)": "C"
    }
    scenario_key = st.sidebar.radio(
        "Scenario",
        list(scenario_map.keys()),
        index=0
    )
    scenario = scenario_map[scenario_key]

    # --- Filter Data ---
    if df_main.empty:
        st.error("No data to display. Check data file paths.")
        st.stop()

    df_filtered = df_main.copy()
    if protocol != "All":
        df_filtered = df_filtered[df_filtered['protocol'] == protocol.lower()]
    if scenario != "All":
        df_filtered = df_filtered[df_filtered['scenario'] == scenario]

    if df_filtered.empty:
        st.warning("No data found for the selected filters.")
    else:
        # --- Top Row KPIs ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Avg. RTT", f"{df_filtered['avg_rtt_ms'].mean():.1f} ms")
        kpi2.metric("Avg. Throughput", f"{df_filtered['avg_bps'].mean() / 1e6:.1f} Mbps")
        kpi3.metric("Packet Loss", f"{df_filtered['packet_loss_rate'].mean():.2f} %")

        congested_flows = df_filtered[df_filtered['congestion_level'] == 'high'].shape[0]
        total_flows = df_filtered.shape[0]
        if total_flows > 0:
            kpi4.metric("Congested Flows", f"{congested_flows} / {total_flows}", f"{congested_flows/total_flows*100:.1f}%")
        else:
            kpi4.metric("Congested Flows", "N/A")

        st.markdown("---")

        # --- Main Dashboard Columns ---
        col1, col2 = st.columns([0.6, 0.4])

        # --- Column 1: Digital Twin View ---
        with col1:
            st.subheader("Digital Twin Topology")
            topology_graph = create_topology_graph(data['topology'])
            if topology_graph:
                st.graphviz_chart(topology_graph)
            
            st.subheader("Analyze Congested Flows")
            df_hotspots = df_filtered[df_filtered['congestion_level'] == 'high'].sort_values('packet_loss_rate', ascending=False)
            
            if df_hotspots.empty:
                st.success("No congested flows found for the selected filters.")
            else:
                # Create a human-readable list for the selectbox
                flow_list = df_hotspots.apply(
                    lambda r: f"{r['client_name']} -> {r['server_name']} (Sample {r['sample_id']}, Loss: {r['packet_loss_rate']:.2f}%)",
                    axis=1
                ).tolist()
                
                selected_flow_str = st.selectbox(
                    "Select a congested flow to analyze:",
                    ["Select a flow..."] + flow_list
                )

        # --- Column 2: ML Brain View ---
        with col2:
            st.subheader("ML Analysis & Optimization")
            
            if selected_flow_str == "Select a flow..." or 'models' not in locals():
                col2.info("Select a congested flow from the list to run ML analysis.")
            else:
                # Find the selected row
                sample_id = int(selected_flow_str.split(" (Sample ")[1].split(",")[0])
                flow_data = df_hotspots[df_hotspots['sample_id'] == sample_id].iloc[0]

                st.markdown(f"**Analyzing Flow:** `{flow_data['client_name']}` -> `{flow_data['server_name']}`")

                # 1. Predicted Cause (Scenario)
                if models['scenario']:
                    try:
                        input_scenario = flow_data[FEATURES_SCENARIO]
                        pred_scenario = models['scenario'].predict([input_scenario])[0]
                        st.metric("Predicted Cause (Scenario)", f"Scenario {pred_scenario}")
                        if pred_scenario == 'B':
                            st.warning("Model indicates this is a local congestion hotspot.", icon="🔥")
                        elif pred_scenario == 'C':
                            st.error("Model indicates this is a BGP failover event.", icon="⚡")
                        else:
                            st.info("Model indicates normal operation (high baseline traffic).", icon="✅")
                    except Exception as e:
                        st.error(f"Scenario model error: {e}")

                st.markdown("---")

                # 2. Performance (Actual vs. Predicted)
                if models['rtt'] and models['throughput']:
                    st.markdown("**Performance Analysis**")
                    try:
                        # RTT
                        input_rtt = flow_data[FEATURES_RTT]
                        pred_rtt = models['rtt'].predict([input_rtt])[0]
                        st.metric("RTT (Actual vs. Predicted)", f"{flow_data['avg_rtt_ms']:.1f} ms", f"{pred_rtt - flow_data['avg_rtt_ms']:.1f} ms (Pred. {pred_rtt:.1f})")
                        
                        # Throughput
                        input_thr = flow_data[FEATURES_THROUGHPUT]
                        pred_thr = models['throughput'].predict([input_thr])[0]
                        st.metric("Throughput (Actual vs. Predicted)", f"{flow_data['avg_bps']/1e6:.1f} Mbps", f"{(pred_thr - flow_data['avg_bps'])/1e6:.1f} Mbps (Pred. {pred_thr/1e6:.1f})")
                    except Exception as e:
                        st.error(f"Performance model error: {e}")

                st.markdown("---")

                # 3. Optimization Recommendation
                if models['buffer']:
                    st.markdown("**Optimization Recommendation**")
                    try:
                        input_buf = flow_data[FEATURES_BUFFER]
                        pred_buf = models['buffer'].predict([input_buf])[0]
                        
                        col_buf1, col_buf2 = st.columns(2)
                        col_buf1.metric("Current Buffer Size", f"{flow_data['server_buffer_bytes']/1024:.0f} KB")
                        col_buf2.metric("Recommended Buffer Size", f"{pred_buf/1024:.0f} KB")
                        
                        if pred_buf > (flow_data['server_buffer_bytes'] * 1.1): # Recommend if >10% increase
                            st.success(f"Recommendation: Increase buffer to {pred_buf/1024:.0f} KB to optimize this flow.", icon="🚀")
                        else:
                            st.info("Current buffer size is sufficient.", icon="✅")
                    except Exception as e:
                        st.error(f"Buffer model error: {e}")


# --- TAB 2: ML MODEL INSIGHTS ---
with tab2:
    st.header("ML Model Insights: Feature Importance")
    st.markdown("""
    These plots show which network metrics were most important for each of our 5 ML models.
    This helps us validate that the models learned real, physical network behavior.
    """)

    plot_paths = [
        "./models_combined/fi_congestion.png",
        "./models_combined/fi_scenario.png",
        "./models_combined/fi_rtt.png",
        "./models_combined/fi_throughput.png",
        "./models_combined/fi_buffer.png"
    ]

    for path_str in plot_paths:
        path = Path(path_str)
        if path.exists():
            st.image(str(path), use_column_width=True)
        else:
            st.warning(f"Plot not found: {path_str}")