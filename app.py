import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import shap
from flaml import AutoML
from geoshapley import GeoShapleyExplainer
import pickle

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GeoShapley Interactive Analysis Tool",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF0051;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 GeoShapley Interactive Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Select Function",
        ["Data Upload", "Model Training", "GeoShapley Analysis", "Results Visualization"],
        index=0
    )
    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Upload CSV data with spatial coordinates
    2. Select features and target variable
    3. Train machine learning model
    4. Run GeoShapley analysis
    5. View visualization results
    """)

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'X_geo' not in st.session_state:
    st.session_state.X_geo = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X_sample' not in st.session_state:
    st.session_state.X_sample = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'svc' not in st.session_state:
    st.session_state.svc = None
if 'svc_features' not in st.session_state:
    st.session_state.svc_features = None
if 'coord_cols' not in st.session_state:
    st.session_state.coord_cols = None
if 'svc_coords' not in st.session_state:
    st.session_state.svc_coords = None
if 'svc_coord_cols' not in st.session_state:
    st.session_state.svc_coord_cols = None

# Page 1: Data Upload
if page == "Data Upload":
    st.markdown('<h2 class="sub-header">📤 Data Upload & Configuration</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Please upload a CSV file containing spatial coordinate data"
    )
    
    # Example data option
    use_example = st.checkbox("Use Example Data (Seattle House Price Data)")
    
    if use_example:
        try:
            @st.cache_data
            def load_example_data():
                url = "https://raw.githubusercontent.com/Ziqi-Li/geoshapley/refs/heads/main/data/seattle_sample_1k.csv"
                return pd.read_csv(url)
            
            st.session_state.data = load_example_data()
            st.success("✅ Example data loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load example data: {str(e)}")
    
    elif uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read data: {str(e)}")
    
    # Display data
    if st.session_state.data is not None:
        st.markdown("### 📊 Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.markdown(f"**Data Shape:** {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
        
        # Data configuration
        st.markdown("### ⚙️ Data Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            # Select target variable
            target_col = st.selectbox(
                "Select Target Variable (y)",
                options=st.session_state.data.columns.tolist(),
                help="Select the target variable to predict"
            )
        
        with col2:
            # Select task type
            task_type = st.selectbox(
                "Task Type",
                ["Regression", "Classification"],
                help="Select machine learning task type"
            )
        
        # Select feature columns
        feature_cols = st.multiselect(
            "Select Feature Columns (excluding spatial coordinates)",
            options=[col for col in st.session_state.data.columns if col != target_col],
            help="Select feature columns for prediction"
        )
        
        # Select spatial coordinate columns
        st.markdown("### 📍 Spatial Coordinate Configuration")
        coord_cols = st.multiselect(
            "Select Spatial Coordinate Columns (must be placed last)",
            options=[col for col in st.session_state.data.columns if col != target_col],
            default=[],
            help="Usually select 2 columns (e.g., longitude, latitude or UTM_X, UTM_Y)"
        )
        
        if len(coord_cols) > 0:
            st.info(f"Selected {len(coord_cols)} spatial coordinate column(s): {', '.join(coord_cols)}")
        
        # Prepare data button
        if st.button("✅ Prepare Data", type="primary", use_container_width=True):
            if len(feature_cols) == 0:
                st.error("Please select at least one feature column!")
            elif len(coord_cols) == 0:
                st.error("Please select at least one spatial coordinate column!")
            else:
                try:
                    # Prepare X_geo (features + spatial coordinates, coordinates must be last)
                    X_features = st.session_state.data[feature_cols].copy()
                    X_coords = st.session_state.data[coord_cols].copy()
                    st.session_state.X_geo = pd.concat([X_features, X_coords], axis=1)
                    st.session_state.y = st.session_state.data[target_col].copy()
                    st.session_state.task_type = task_type
                    st.session_state.coord_cols = coord_cols  # Save coordinate column names
                    
                    st.success(f"✅ Data preparation completed! Features: {len(feature_cols)}, Spatial coordinates: {len(coord_cols)}")
                    st.dataframe(st.session_state.X_geo.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Data preparation failed: {str(e)}")

# Page 2: Model Training
elif page == "Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.X_geo is None:
        st.warning("⚠️ Please prepare data in the 'Data Upload' page first!")
    else:
        st.markdown("### 📊 Data Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(st.session_state.X_geo))
        with col2:
            st.metric("Features", len(st.session_state.X_geo.columns) - len(st.session_state.X_geo.columns[-2:]))
        with col3:
            st.metric("Spatial Coordinates", len(st.session_state.X_geo.columns[-2:]))
        
        # Model selection
        st.markdown("### 🎯 Model Selection")
        model_option = st.selectbox(
            "Select Model Type",
            ["AutoML (FLAML)", "Random Forest", "Neural Network (MLP)"],
            help="AutoML will automatically select the best model"
        )
        
        # Training parameters
        st.markdown("### ⚙️ Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random Seed", 0, 1000, 42)
        
        with col2:
            if model_option == "AutoML (FLAML)":
                time_budget = st.number_input("Time Budget (seconds)", 10, 600, 60, 10)
            elif model_option == "Random Forest":
                n_estimators = st.number_input("Number of Trees", 10, 500, 100, 10)
            elif model_option == "Neural Network (MLP)":
                hidden_layers = st.text_input("Hidden Layer Structure", "100,50", help="Separated by commas, e.g.: 100,50")
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                with st.spinner("Training model..."):
                    # Data splitting
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X_geo,
                        st.session_state.y,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Train model
                    if model_option == "AutoML (FLAML)":
                        automl = AutoML()
                        settings = {
                            "time_budget": time_budget,
                            "metric": 'r2' if st.session_state.task_type == "Regression" else 'accuracy',
                            "task": 'regression' if st.session_state.task_type == "Regression" else 'classification',
                            "eval_method": 'cv',
                            "n_splits": 5,
                            "verbose": 0
                        }
                        automl.fit(X_train, y_train, **settings)
                        model = automl.model.estimator
                        st.session_state.model_type = "automl"
                    elif model_option == "Random Forest":
                        if st.session_state.task_type == "Regression":
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                        else:
                            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                        model.fit(X_train, y_train)
                        st.session_state.model_type = "rf"
                    elif model_option == "Neural Network (MLP)":
                        hidden_layer_sizes = tuple([int(x) for x in hidden_layers.split(',')])
                        if st.session_state.task_type == "Regression":
                            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=500)
                        else:
                            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=500)
                        model.fit(X_train, y_train)
                        st.session_state.model_type = "mlp"
                    
                    # Evaluate model
                    y_pred = model.predict(X_test)
                    if st.session_state.task_type == "Regression":
                        score = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        st.metric("R² Score", f"{score:.4f}")
                        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                    else:
                        score = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", f"{score:.4f}")
                    
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.success("✅ Model training completed!")
                    
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.exception(e)

# Page 3: GeoShapley Analysis
elif page == "GeoShapley Analysis":
    st.markdown('<h2 class="sub-header">🔍 GeoShapley Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model in the 'Model Training' page first!")
    else:
        st.markdown("### ⚙️ Analysis Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            background_method = st.selectbox(
                "Background Data Method",
                ["K-means Clustering", "Random Sampling"],
                help="K-means clustering usually works better"
            )
            background_size = st.number_input("Background Data Size", 10, 200, 100, 10)
        
        with col2:
            n_jobs = st.number_input("Number of Parallel Jobs", -1, 8, -1, help="-1 means using all CPU cores")
            g = st.number_input("Number of Spatial Coordinates (g)", 1, 5, 2, help="Usually 2 (longitude, latitude)")
        
        # Analysis button
        if st.button("🔬 Start Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("Running GeoShapley analysis, this may take some time..."):
                    # Prepare background data
                    if background_method == "K-means Clustering":
                        background_X = shap.kmeans(st.session_state.X_train, k=min(background_size, len(st.session_state.X_train))).data
                    else:
                        background_X = st.session_state.X_train.sample(min(background_size, len(st.session_state.X_train))).values
                    
                    # Create explainer
                    explainer = GeoShapleyExplainer(
                        st.session_state.model.predict,
                        background_X,
                        g=g
                    )
                    
                    # Explain data (use smaller sample for faster processing)
                    sample_size = st.slider("Analysis Sample Size", 10, len(st.session_state.X_geo), min(100, len(st.session_state.X_geo)), 10)
                    X_sample = st.session_state.X_geo.sample(n=min(sample_size, len(st.session_state.X_geo)), random_state=42)
                    
                    results = explainer.explain(X_sample, n_jobs=n_jobs)
                    
                    st.session_state.results = results
                    st.session_state.X_sample = X_sample
                    st.success("✅ GeoShapley analysis completed!")
                    
                    # Display basic statistics
                    st.markdown("### 📈 Basic Statistics")
                    summary_stats = results.summary_statistics()
                    st.dataframe(summary_stats, use_container_width=True)
                    
                    # Check additivity
                    st.markdown("### ✅ Additivity Check")
                    total = np.sum(results.primary, axis=1) + results.geo + np.sum(results.geo_intera, axis=1)
                    is_additive = np.allclose(total + results.base_value, 
                                             results.predict_f(X_sample).reshape(-1), atol=1e-5)
                    if is_additive:
                        st.success("✅ Components add up to model prediction: True")
                    else:
                        st.warning("⚠️ Components do not add up to model prediction: False")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)

# Page 4: Results Visualization
elif page == "Results Visualization":
    st.markdown('<h2 class="sub-header">📊 Results Visualization</h2>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("⚠️ Please run analysis in the 'GeoShapley Analysis' page first!")
    else:
        results = st.session_state.results
        
        # Visualization options
        st.markdown("### 🎨 Visualization Options")
        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Summary Plot", "Partial Dependence Plot", "Contribution Bar Plot", "Summary Statistics Table"]
        )
        
        if viz_option == "Summary Plot":
            st.markdown("#### 📊 SHAP-style Summary Plot")
            include_interaction = st.checkbox("Include Interaction Effects", value=True)
            dpi = st.slider("Image Resolution (DPI)", 50, 300, 120)
            
            if st.button("Generate Summary Plot", type="primary"):
                try:
                    # summary_plot already creates a figure internally, no need to create one externally
                    results.summary_plot(include_interaction=include_interaction, dpi=dpi)
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Failed to generate plot: {str(e)}")
                    st.exception(e)
        
        elif viz_option == "Partial Dependence Plot":
            st.markdown("#### 📈 Partial Dependence Plot")
            col1, col2 = st.columns(2)
            with col1:
                max_cols = st.number_input("Max Columns per Row", 2, 6, 3)
                gam_curve = st.checkbox("Show GAM Smoothing Curve", value=False)
            with col2:
                dpi = st.slider("Image Resolution (DPI)", 50, 300, 150)
            
            if st.button("Generate Partial Dependence Plot", type="primary"):
                try:
                    results.partial_dependence_plots(
                        max_cols=max_cols,
                        gam_curve=gam_curve,
                        dpi=dpi
                    )
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Failed to generate plot: {str(e)}")
                    if "pygam" in str(e):
                        st.info("💡 Tip: To use GAM smoothing curve, install pygam: pip install pygam")
        
        elif viz_option == "Contribution Bar Plot":
            st.markdown("#### 📊 Feature Contribution Ranking Plot")
            dpi = st.slider("Image Resolution (DPI)", 50, 300, 150)
            
            if st.button("Generate Contribution Bar Plot", type="primary"):
                try:
                    results.contribution_bar_plot(dpi=dpi)
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Failed to generate plot: {str(e)}")
        
        elif viz_option == "Summary Statistics Table":
            st.markdown("#### 📋 Summary Statistics Table")
            include_interaction = st.checkbox("Include Interaction Effects", value=True)
            
            summary_stats = results.summary_statistics(include_interaction=include_interaction)
            st.dataframe(summary_stats, use_container_width=True)
            
            # Download button
            csv = summary_stats.to_csv().encode('utf-8-sig')
            st.download_button(
                label="📥 Download Summary Statistics (CSV)",
                data=csv,
                file_name="geoshapley_summary_stats.csv",
                mime="text/csv"
            )
        
        # Spatially Varying Coefficients
        st.markdown("---")
        st.markdown("### 🗺️ Spatially Varying Coefficients (SVC)")
        
        # Check if coordinate data is available
        if st.session_state.X_sample is not None and len(st.session_state.X_sample.columns) >= 2:
            # Get coordinate columns (last g columns)
            g = results.g
            coord_cols = st.session_state.X_sample.columns[-g:].tolist()
            
            st.markdown("#### ⚙️ SVC Parameter Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                # Select features to calculate SVC
                feature_names = st.session_state.X_sample.columns[:-g].tolist()
                selected_features = st.multiselect(
                    "Select Features for SVC Calculation",
                    options=feature_names,
                    default=feature_names[:min(3, len(feature_names))],
                    help="Select features to calculate spatially varying coefficients"
                )
                
                coef_type = st.selectbox(
                    "Coefficient Type",
                    ["raw", "gwr"],
                    help="raw: Raw coefficients (may contain extreme values); gwr: GWR-smoothed coefficients (requires mgwr package)"
                )
            
            with col2:
                include_primary = st.checkbox("Include Primary Effects", value=False, 
                                             help="Whether to include primary effects in spatial coefficients")
                
                if coef_type == "gwr":
                    st.info("💡 GWR method requires mgwr package: pip install mgwr")
            
            if st.button("🔬 Calculate SVC", type="primary"):
                try:
                    with st.spinner("Calculating spatially varying coefficients..."):
                        # Get feature column indices
                        feature_indices = [feature_names.index(f) for f in selected_features]
                        
                        # Get coordinates
                        coords = st.session_state.X_sample[coord_cols].values
                        
                        # Calculate SVC
                        if coef_type == "gwr":
                            try:
                                import mgwr
                            except ImportError:
                                st.error("❌ mgwr package not installed! Please run: pip install mgwr")
                                st.stop()
                        
                        svc = results.get_svc(
                            col=feature_indices,
                            coef_type=coef_type,
                            include_primary=include_primary,
                            coords=coords
                        )
                        
                        # Ensure svc is 2D array
                        if svc.ndim == 1:
                            svc = svc.reshape(-1, 1)
                        
                        st.session_state.svc = svc
                        st.session_state.svc_features = selected_features
                        st.session_state.svc_coords = coords
                        st.session_state.svc_coord_cols = coord_cols
                        st.success(f"✅ SVC calculation completed! Calculated spatially varying coefficients for {len(selected_features)} feature(s)")
                        
                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")
                    st.exception(e)
            
            # Display SVC results and visualizations if SVC has been calculated
            if st.session_state.svc is not None and st.session_state.svc_features is not None:
                svc = st.session_state.svc
                svc_features = st.session_state.svc_features
                coords = st.session_state.svc_coords
                coord_cols = st.session_state.svc_coord_cols
                
                # Display SVC statistics
                st.markdown("#### 📊 SVC Statistics")
                svc_df = pd.DataFrame(svc, columns=svc_features)
                st.dataframe(svc_df.describe(), use_container_width=True)
                
                # Visualization options
                st.markdown("#### 🎨 SVC Visualization")
                viz_type = st.selectbox(
                    "Select Visualization Type",
                    ["Scatter Plot", "Heatmap", "Statistical Plot"],
                    key="svc_viz_type"
                )
                
                if viz_type == "Scatter Plot":
                    # Create scatter plots for each feature
                    num_features = len(svc_features)
                    num_cols = min(3, num_features)
                    num_rows = (num_features + num_cols - 1) // num_cols
                    
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
                    if num_rows == 1:
                        axes = [axes] if num_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for idx, feature in enumerate(svc_features):
                        ax = axes[idx]
                        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                                             c=svc[:, idx], cmap='viridis', 
                                             s=30, alpha=0.6)
                        ax.set_xlabel(coord_cols[0])
                        ax.set_ylabel(coord_cols[1])
                        ax.set_title(f'Spatially Varying Coefficient: {feature}')
                        plt.colorbar(scatter, ax=ax)
                    
                    # Hide extra subplots
                    for idx in range(num_features, len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                elif viz_type == "Heatmap":
                    # Create heatmap matrix
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(svc.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
                    ax.set_yticks(range(len(svc_features)))
                    ax.set_yticklabels(svc_features)
                    ax.set_xlabel('Sample Index')
                    ax.set_title('Spatially Varying Coefficients Heatmap')
                    plt.colorbar(im, ax=ax, label='SVC Value')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                elif viz_type == "Statistical Plot":
                    # Create box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    svc_df.boxplot(ax=ax, rot=45)
                    ax.set_ylabel('SVC Value')
                    ax.set_title('Spatially Varying Coefficients Distribution')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Download SVC data
                st.markdown("#### 📥 Download SVC Data")
                csv = svc_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="Download SVC Data (CSV)",
                    data=csv,
                    file_name="svc_coefficients.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please complete GeoShapley analysis first to calculate spatially varying coefficients")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>GeoShapley - A Game Theory Approach to Measuring Spatial Effects</p>
    <p>Powered by Streamlit | Made with ❤️</p>
</div>
""", unsafe_allow_html=True)

