from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Ensure repo root is on sys.path so src/ imports work when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum_model import train_quantum_model   # quantum model wrapper
from src.classical_model import (
    train_classical_model,
    classical_portfolio_opt,
)  # classical baselines
from src.portfolio_qaoa import portfolio_qubo


@st.cache_data(show_spinner=False)
def generate_synthetic_nonlinear_data(n_samples=10000, n_features=10, random_state=42, use_saved=True):
    """
    Generate a synthetic non-linear dataset designed to favor quantum models.
    Creates complex non-linear patterns that quantum feature maps can capture better.
    
    If use_saved=True and the CSV file exists, loads from file for consistency.
    Otherwise generates new data.
    """
    # Check if saved dataset exists
    if use_saved:
        saved_path = REPO_ROOT / "data" / "synthetic_nonlinear_dataset.csv"
        if saved_path.exists():
            try:
                df = pd.read_csv(saved_path)
                if df.shape[0] == n_samples and df.shape[1] == n_features + 1:
                    return df
            except Exception:
                pass  # If loading fails, generate new data
    
    # Generate new data if file doesn't exist or loading failed
    np.random.seed(random_state)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear target with complex patterns
    # Use trigonometric, polynomial, and interaction terms
    y = np.zeros(n_samples)
    
    # Pre-compute some global statistics for threshold
    all_scores = []
    
    for i in range(n_samples):
        x = X[i]
        
        # Complex non-linear combination:
        # - Trigonometric terms (quantum-inspired - these are key!)
        # - Polynomial interactions
        # - Circular/spiral patterns
        # - XOR-like patterns
        
        # Circular pattern: distance from origin
        radius = np.sqrt(np.sum(x[:3]**2))
        angle = np.arctan2(x[1], x[0]) if x[0] != 0 else 0
        
        # Trigonometric patterns (quantum models excel at these!)
        trig_term = np.sin(radius * 2) * np.cos(angle * 3) + np.sin(x[0] * x[1])
        
        # Polynomial interactions (high-order)
        poly_term = (x[0]**2 * x[1]) + (x[2]**3) - (x[0] * x[1] * x[2]) + (x[0]**2 * x[2]**2)
        
        # XOR-like pattern (non-linearly separable - hard for linear models)
        xor_term = np.sign(x[3] * x[4]) * np.abs(x[3] - x[4]) + np.sign(x[5] * x[6])
        
        # Spiral/helical pattern
        spiral = np.sin(radius + angle * 2) * np.exp(-radius/4)
        
        # Additional quantum-friendly patterns
        quantum_pattern = np.cos(x[0] * np.pi) * np.sin(x[1] * np.pi) * np.cos(x[2] * np.pi)
        
        # High-order interactions
        high_order = (x[7] * x[8] * x[9])**2 - np.abs(x[7] + x[8] - x[9])
        
        # Combine all non-linear terms (weighted to favor quantum patterns)
        score = (
            3.0 * trig_term +        # Strong weight on trigonometric (quantum strength)
            2.0 * quantum_pattern +   # Quantum-specific patterns
            1.5 * poly_term +
            1.2 * xor_term +
            1.0 * spiral +
            0.8 * high_order +
            0.3 * np.sum(x[5:]**2) - 0.2 * np.sum(x[:5])
        )
        
        all_scores.append(score)
    
    # Use median as threshold for balanced classes
    threshold = np.median(all_scores)
    y = (np.array(all_scores) > threshold).astype(int)
    
    # Add small amount of noise to make it more realistic
    noise = np.random.randn(n_samples) * 0.2
    y = ((np.array(all_scores) + noise) > threshold).astype(int)
    
    # Create DataFrame for compatibility
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to file for future use (if directory exists)
    if use_saved:
        try:
            saved_path = REPO_ROOT / "data" / "synthetic_nonlinear_dataset.csv"
            saved_path.parent.mkdir(exist_ok=True)
            df.to_csv(saved_path, index=False)
        except Exception:
            pass  # If saving fails, just return the data
    
    return df


@st.cache_data(show_spinner=False)
def preprocess_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load or generate synthetic non-linear dataset and perform preprocessing.
    Returns scaled train/test splits.
    """
    # Load or generate synthetic non-linear dataset (uses saved CSV if available)
    df = generate_synthetic_nonlinear_data(n_samples=10000, n_features=10, random_state=random_state, use_saved=True)

    # ----- FEATURE/TARGET SPLIT -----
    X = df.drop(columns=['target'])
    y = df['target']

    # ----- TRAIN-TEST SPLIT -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ----- STANDARDIZATION -----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, df


def render_metrics_block(title: str, metrics: dict, extra_keys=None, color="blue"):
    """Nicely render accuracy and classification report with visualizations."""
    if extra_keys is None:
        extra_keys = []

    # Color-coded header
    if "Quantum" in title:
        header_color = "#8B5CF6"  # Purple for quantum
        icon = "⚛️"
    else:
        header_color = "#3B82F6"  # Blue for classical
        icon = "🧠"
    
    st.markdown(f"<h3 style='color: {header_color};'>{icon} {title}</h3>", unsafe_allow_html=True)
    
    # Main accuracy metric with visual bar
    if "accuracy" in metrics:
        acc = metrics['accuracy']
        delta_color = "normal"
        if acc >= 0.85:
            delta_color = "normal"
        elif acc >= 0.75:
            delta_color = "normal"
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Accuracy", f"{acc:.1%}", delta=f"{acc-0.5:.1%} vs baseline")
        with col2:
            st.progress(acc, text=f"{acc:.1%}")

    # Additional metrics in columns
    if extra_keys:
        cols = st.columns(len(extra_keys))
        for i, key in enumerate(extra_keys):
            if key in metrics and isinstance(metrics[key], (int, float)):
                with cols[i]:
                    st.metric(key.upper(), f"{metrics[key]:.3f}")

    # Classification report as a styled table
    report = metrics.get("report")
    if isinstance(report, dict):
        report_df = (
            pd.DataFrame(report)
            .T.reset_index()
            .rename(columns={"index": "Class"})
        )
        # Remove accuracy row if present
        report_df = report_df[report_df['Class'] != 'accuracy']
        
        # Style the dataframe
        st.markdown("**Classification Report:**")
        st.dataframe(
            report_df.style.format({
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1-score': '{:.3f}',
                'support': '{:.0f}'
            }).background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlGn'),
            use_container_width=True,
            hide_index=True
        )


def main():
    st.set_page_config(
        page_title="Quantum ML Dashboard",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; color: white;">⚛️ Quantum vs Classical ML Dashboard</h1>
        <p style="margin:0.5rem 0 0 0; opacity: 0.9;">Compare quantum-inspired models with classical baselines</p>
    </div>
    """, unsafe_allow_html=True)

    # ----- SIDEBAR CONTROLS -----
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # Mode selection with icons
    mode = st.sidebar.radio(
        "**Select Application**", 
        ["💳 Credit Risk Modeling", "📊 Portfolio Optimization"], 
        index=0
    )
    
    # Extract mode name
    if "Credit Risk" in mode:
        mode = "Credit Risk Modeling"
    else:
        mode = "Portfolio Optimization"
    
    st.sidebar.markdown("---")
    
    run_button = None
    run_portfolio = None
    n_assets = 6
    cardinality = 3
    risk_aversion = 0.1
    
    if mode == "Credit Risk Modeling":
        st.sidebar.markdown("### 📈 Model Settings")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05, help="Fraction of data to use for testing")
        random_state = st.sidebar.number_input("Random Seed", value=42, step=1, help="For reproducibility")
        run_button = st.sidebar.button("🚀 Run Training", type="primary", use_container_width=True)
    elif mode == "Portfolio Optimization":
        st.sidebar.markdown("### 📊 Portfolio Settings")
        n_assets = st.sidebar.number_input(
            "Number of Assets", 
            min_value=4, max_value=10, value=6, step=1, 
            key="sidebar_n_assets",
            help="Total number of assets in the portfolio"
        )
        cardinality = st.sidebar.number_input(
            "Max Assets to Pick", 
            min_value=1, max_value=10, value=3, step=1, 
            key="sidebar_cardinality",
            help="Maximum number of assets to select"
        )
        risk_aversion = st.sidebar.slider(
            "Risk Aversion (λ)", 
            0.01, 1.0, 0.1, 0.01, 
            key="sidebar_risk",
            help="Higher values penalize risk more"
        )
        run_portfolio = st.sidebar.button("🚀 Run Portfolio Optimization", type="primary", key="sidebar_portfolio", use_container_width=True)

    # ----- DATA PREVIEW -----
    with st.expander("Show raw data sample", expanded=False):
        try:
            df = generate_synthetic_nonlinear_data(n_samples=10000, n_features=10, random_state=42, use_saved=True)
            st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            saved_path = REPO_ROOT / "data" / "synthetic_nonlinear_dataset.csv"
            if saved_path.exists():
                st.info(f"📁 Loaded from: `data/synthetic_nonlinear_dataset.csv`")
            else:
                st.info("🔄 Generated on-the-fly (will be saved for next time)")
            st.write("**Synthetic Non-Linear Dataset** - Designed with complex patterns that favor quantum models")
            st.dataframe(df.head(), width='stretch')
        except Exception as e:
            st.error(f"Error loading/generating data: {e}")

    # ==========================
    # CREDIT RISK CLASSIFICATION
    # ==========================
    if mode == "Credit Risk Modeling":
        st.markdown("## 💳 Non-Linear Classification")
        st.markdown("Compare quantum-inspired and classical models on a **synthetic non-linear dataset** designed to showcase quantum advantages")
        
        if not run_button:
            st.info("👆 **Click 'Run Training' in the sidebar** to train and compare models.")
            
            # Show dataset info
            try:
                df = generate_synthetic_nonlinear_data(n_samples=10000, n_features=10, random_state=42, use_saved=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Features", df.shape[1] - 1)  # Exclude target
                with col3:
                    if "target" in df.columns:
                        class_1_rate = df["target"].mean()
                        st.metric("Class 1 Rate", f"{class_1_rate:.1%}")
            except:
                pass
        else:
            # ----- TRAINING PIPELINE -----
            try:
                with st.spinner("🔄 Preprocessing data..."):
                    X_train, X_test, y_train, y_test, df_clean = preprocess_data(
                        test_size=test_size, random_state=int(random_state)
                    )
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")
                return

            q_metrics = None
            c_metrics = None
            
            left, right = st.columns(2)

            with left:
                try:
                    # Create progress tracking for quantum model
                    q_progress_container = st.empty()
                    q_status_container = st.empty()
                    
                    def q_progress_callback(progress, message):
                        q_progress_container.progress(progress, text=message)
                        q_status_container.text(f"⚛️ {message}")
                    
                    # Show initial status
                    q_status_container.text("⚛️ Starting quantum model training...")
                    q_progress_container.progress(0, text="Initializing...")
                    
                    # Try with progress_callback, fallback without it
                    try:
                        _, _, q_metrics = train_quantum_model(
                            X_train, y_train, X_test, y_test, 
                            progress_callback=q_progress_callback
                        )
                    except TypeError:
                        # Fallback if progress_callback not supported
                        q_status_container.text("⚛️ Training quantum model...")
                        _, _, q_metrics = train_quantum_model(
                            X_train, y_train, X_test, y_test
                        )
                    except Exception as e:
                        q_status_container.error(f"Error: {e}")
                        raise
                    
                    q_progress_container.empty()
                    q_status_container.empty()
                    render_metrics_block("Quantum Model", q_metrics, color="purple")
                except Exception as e:
                    st.error(f"❌ Quantum model training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            with right:
                try:
                    # Create progress tracking for classical model
                    c_progress_container = st.empty()
                    c_status_container = st.empty()
                    
                    def c_progress_callback(progress, message):
                        c_progress_container.progress(progress, text=message)
                        c_status_container.text(f"🧠 {message}")
                    
                    # Try with progress_callback, fallback without it
                    try:
                        _, _, c_metrics = train_classical_model(
                            X_train, y_train, X_test, y_test,
                            progress_callback=c_progress_callback
                        )
                    except TypeError:
                        # Fallback if progress_callback not supported
                        c_status_container.text("🧠 Training classical model (no progress tracking)...")
                        _, _, c_metrics = train_classical_model(
                            X_train, y_train, X_test, y_test
                        )
                    
                    c_progress_container.empty()
                    c_status_container.empty()
                    render_metrics_block("Classical Model", c_metrics, extra_keys=["auc"], color="blue")
                except Exception as e:
                    st.error(f"❌ Classical model training failed: {e}")

            st.success("✅ Training complete!")
            
            # Comparison visualization
            if q_metrics and c_metrics:
                st.markdown("---")
                st.markdown("### 📊 Model Comparison")
                
                # Comparison chart
                comparison_data = {
                    'Model': ['Quantum', 'Classical'],
                    'Accuracy': [q_metrics['accuracy'], c_metrics['accuracy']]
                }
                if 'auc' in c_metrics:
                    comparison_data['AUC'] = [q_metrics.get('auc', 0), c_metrics['auc']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Quantum',
                    x=['Accuracy'],
                    y=[q_metrics['accuracy']],
                    marker_color='#8B5CF6',
                    text=[f"{q_metrics['accuracy']:.1%}"],
                    textposition='auto',
                ))
                fig.add_trace(go.Bar(
                    name='Classical',
                    x=['Accuracy'],
                    y=[c_metrics['accuracy']],
                    marker_color='#3B82F6',
                    text=[f"{c_metrics['accuracy']:.1%}"],
                    textposition='auto',
                ))
                fig.update_layout(
                    title="Accuracy Comparison",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1]),
                    barmode='group',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Winner badge
                if q_metrics['accuracy'] > c_metrics['accuracy']:
                    st.success("🏆 **Quantum model performs better!**")
                elif c_metrics['accuracy'] > q_metrics['accuracy']:
                    st.info("🏆 **Classical model performs better!**")
                else:
                    st.info("🤝 **Models perform equally!**")

    # ==========================
    # PORTFOLIO OPTIMIZATION DEMO
    # ==========================
    elif mode == "Portfolio Optimization":
        st.markdown("## 📊 Portfolio Optimization")
        st.markdown("Compare **classical brute-force** Markowitz optimizer with **QAOA-based** quantum optimizer")

        if not run_portfolio:
            st.info("👆 **Adjust settings in the sidebar and click 'Run Portfolio Optimization'** to compare methods.")
            
            # Show parameter summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Assets Available", n_assets)
            with col2:
                st.metric("Max to Select", cardinality)
            with col3:
                st.metric("Risk Aversion (λ)", f"{risk_aversion:.2f}")
        else:
            # Synthetic returns and covariance
            np.random.seed(0)
            n = int(n_assets)
            mu = np.random.normal(0.05, 0.02, size=n)
            A = np.random.rand(n, n)
            cov = A.T @ A * 0.001

            progress = st.progress(0)
            status = st.empty()

            status.text("🔄 Running classical brute-force optimizer...")
            progress.progress(30)
            best_classical, val_classical = None, None
            try:
                best_classical, val_classical = classical_portfolio_opt(
                    mu, cov, cardinality=int(cardinality), risk_aversion=float(risk_aversion)
                )
            except Exception as e:
                st.error(f"❌ Classical portfolio optimization failed: {e}")

            status.text("⚛️ Running QAOA portfolio optimizer...")
            progress.progress(60)
            qres = None
            try:
                qres = portfolio_qubo(
                    mu,
                    cov,
                    cardinality=int(cardinality),
                    risk_aversion=float(risk_aversion),
                    reps=1,
                )
            except Exception as e:
                st.error(f"❌ QAOA portfolio optimization failed: {e}")

            progress.progress(100)
            status.text("✅ Optimization complete!")

            # Display results side by side with visualizations
            ccol, qcol = st.columns(2)
            
            with ccol:
                st.markdown("### 🧠 Classical (Brute Force)")
                if best_classical is not None:
                    selected = best_classical.astype(int)
                    selected_indices = [i for i, x in enumerate(selected) if x > 0.5]
                    
                    # Selection visualization
                    fig_classical = go.Figure(data=[
                        go.Bar(x=[f"Asset {i}" for i in range(n)], 
                              y=selected,
                              marker_color=['#10b981' if i in selected_indices else '#e5e7eb' for i in range(n)],
                              text=selected,
                              textposition='auto')
                    ])
                    fig_classical.update_layout(
                        title="Selected Assets",
                        yaxis_title="Selected (1=Yes, 0=No)",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_classical, use_container_width=True)
                    
                    exp_ret = float(mu @ best_classical)
                    variance = float(best_classical @ cov @ best_classical)
                    
                    st.metric("📈 Score", f"{val_classical:.4f}", help="Expected Return - λ × Variance")
                    st.metric("💰 Expected Return", f"{exp_ret:.4f}")
                    st.metric("📊 Variance", f"{variance:.6f}")
                    st.caption(f"Selected assets: {selected_indices}")
                else:
                    st.warning("No classical solution available.")

            with qcol:
                st.markdown("### ⚛️ Quantum (QAOA)")
                if qres is not None:
                    sel = qres["selection"]
                    selected_indices = [i for i, x in enumerate(sel) if x > 0.5]
                    
                    # Selection visualization
                    fig_quantum = go.Figure(data=[
                        go.Bar(x=[f"Asset {i}" for i in range(n)], 
                              y=sel,
                              marker_color=['#8b5cf6' if i in selected_indices else '#e5e7eb' for i in range(n)],
                              text=sel,
                              textposition='auto')
                    ])
                    fig_quantum.update_layout(
                        title="Selected Assets",
                        yaxis_title="Selected (1=Yes, 0=No)",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_quantum, use_container_width=True)
                    
                    st.metric("📈 Score", f"{qres['score']:.4f}", help="Expected Return - λ × Variance")
                    st.metric("💰 Expected Return", f"{qres['expected_return']:.4f}")
                    st.metric("📊 Variance", f"{qres['variance']:.6f}")
                    st.caption(f"Selected assets: {selected_indices}")
                else:
                    st.warning("No QAOA solution available.")

            # Comparison chart
            if best_classical is not None and qres is not None:
                st.markdown("---")
                st.markdown("### 📊 Performance Comparison")
                
                comparison_fig = go.Figure()
                comparison_fig.add_trace(go.Bar(
                    name='Classical',
                    x=['Score', 'Expected Return', 'Variance'],
                    y=[val_classical, float(mu @ best_classical), float(best_classical @ cov @ best_classical)],
                    marker_color='#3B82F6',
                    text=[f"{val_classical:.4f}", f"{float(mu @ best_classical):.4f}", f"{float(best_classical @ cov @ best_classical):.6f}"],
                    textposition='auto',
                ))
                comparison_fig.add_trace(go.Bar(
                    name='Quantum',
                    x=['Score', 'Expected Return', 'Variance'],
                    y=[qres['score'], qres['expected_return'], qres['variance']],
                    marker_color='#8B5CF6',
                    text=[f"{qres['score']:.4f}", f"{qres['expected_return']:.4f}", f"{qres['variance']:.6f}"],
                    textposition='auto',
                ))
                comparison_fig.update_layout(
                    title="Portfolio Optimization Comparison",
                    barmode='group',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Winner
                if qres['score'] > val_classical:
                    st.success(f"🏆 **Quantum QAOA found a better solution!** (Score: {qres['score']:.4f} vs {val_classical:.4f})")
                elif val_classical > qres['score']:
                    st.info(f"🏆 **Classical method found a better solution!** (Score: {val_classical:.4f} vs {qres['score']:.4f})")
                else:
                    st.info("🤝 **Both methods found equivalent solutions!**")


if __name__ == "__main__":
    # Allow running as a regular script for debugging, though
    # Streamlit is the primary entrypoint.
    main()
