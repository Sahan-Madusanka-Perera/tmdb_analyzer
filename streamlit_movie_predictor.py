
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="TMDB Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# TMDBPredictiveAnalyzer Class (embedded for Streamlit)
class TMDBPredictiveAnalyzer:
    def __init__(self, model_path=None):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def predict_revenue(self, budget, runtime, vote_average, release_month, primary_genre, 
                       vote_count=1000, release_year=2024):
        if not self.model:
            raise ValueError("Model not trained or loaded")

        input_data = self._prepare_input(budget, runtime, vote_average, release_month, 
                                       primary_genre, vote_count, release_year)
        pred_log = self.model.predict(input_data)
        return np.expm1(pred_log[0])

    def analyze_roi(self, budget, runtime, vote_average, release_month, primary_genre,
                   vote_count=1000, release_year=2024):
        predicted_revenue = self.predict_revenue(budget, runtime, vote_average, release_month,
                                                primary_genre, vote_count, release_year)
        roi = (predicted_revenue - budget) / budget * 100

        return {
            'predicted_revenue': predicted_revenue,
            'investment': budget,
            'profit': predicted_revenue - budget,
            'roi_percentage': roi,
            'risk_level': 'Low' if roi > 50 else 'Medium' if roi > 0 else 'High'
        }

    def optimize_release_timing(self, budget, runtime, vote_average, primary_genre,
                               vote_count=1000, release_year=2024):
        monthly_roi = {}
        for month in range(1, 13):
            analysis = self.analyze_roi(budget, runtime, vote_average, month, primary_genre,
                                      vote_count, release_year)
            monthly_roi[month] = analysis['roi_percentage']

        best_month = max(monthly_roi, key=monthly_roi.get)
        return {
            'best_month': best_month,
            'best_roi': monthly_roi[best_month],
            'monthly_roi': monthly_roi
        }

    def _prepare_input(self, budget, runtime, vote_average, release_month, primary_genre,
                      vote_count, release_year):
        input_data = pd.DataFrame({
            'budget': [budget],
            'runtime': [runtime],
            'vote_average': [vote_average],
            'vote_count': [vote_count],
            'release_year': [release_year],
            'release_month': [release_month],
            'primary_genre': [primary_genre]
        })

        # Feature engineering
        input_data['budget_log'] = np.log1p(input_data['budget'])
        input_data['vote_score'] = input_data['vote_average'] * input_data['vote_count']
        input_data['is_summer_release'] = input_data['release_month'].isin([5,6,7,8]).astype(int)
        input_data['is_holiday_release'] = input_data['release_month'].isin([11,12]).astype(int)
        input_data['release_quarter'] = input_data['release_month'].apply(lambda x: (x-1)//3 + 1)

        # Add missing columns with defaults
        for col in self.feature_names:
            if col not in input_data.columns:
                if col.startswith('genre_'):
                    input_data[col] = 1 if col == f'genre_{primary_genre}' else 0
                else:
                    input_data[col] = 0

        return input_data[self.feature_names]

    def load_model(self, path):
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.preprocessor = loaded['preprocessor'] 
        self.feature_names = loaded['feature_names']

# Initialize the analyzer
@st.cache_resource
def load_analyzer():
    import os
    # Try different possible paths for the model
    possible_paths = [
        'models/tmdb_analyzer.pkl',
        './models/tmdb_analyzer.pkl',
        os.path.join(os.getcwd(), 'models', 'tmdb_analyzer.pkl'),
        'tmdb_analyzer.pkl'
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                st.info(f"Loading model from: {model_path}")
                analyzer = TMDBPredictiveAnalyzer(model_path)
                st.success("✅ Model loaded successfully!")
                return analyzer
            except Exception as e:
                st.warning(f"Error loading from {model_path}: {str(e)}")
                continue
    
    # If no model found, show detailed error
    st.error("❌ Model not found in any expected location!")
    st.write("**Checked paths:**")
    for path in possible_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        st.write(f"{exists} {path}")
    st.write("**Current working directory:**", os.getcwd())
    st.write("**Available files:**", os.listdir('.'))
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 TMDB Movie Revenue Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Box Office Prediction & ROI Analysis")
    st.markdown("---")

    # Load analyzer
    analyzer = load_analyzer()
    if not analyzer:
        st.stop()

    # Sidebar for inputs
    st.sidebar.header("🎯 Movie Parameters")
    st.sidebar.markdown("Configure your movie details below:")

    # Input parameters
    budget = st.sidebar.slider(
        "💰 Budget ($)", 
        min_value=100_000, 
        max_value=300_000_000, 
        value=50_000_000, 
        step=1_000_000,
        format="$%d"
    )

    runtime = st.sidebar.slider(
        "⏱️ Runtime (minutes)", 
        min_value=60, 
        max_value=240, 
        value=120
    )

    vote_average = st.sidebar.slider(
        "⭐ Expected Rating (1-10)", 
        min_value=1.0, 
        max_value=10.0, 
        value=7.0, 
        step=0.1
    )

    vote_count = st.sidebar.slider(
        "👥 Expected Vote Count", 
        min_value=100, 
        max_value=10_000, 
        value=2_000, 
        step=100
    )

    primary_genre = st.sidebar.selectbox(
        "🎭 Primary Genre",
        ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
         'Drama', 'Fantasy', 'Horror', 'Romance', 'Thriller']
    )

    release_month = st.sidebar.selectbox(
        "📅 Release Month",
        list(range(1, 13)),
        index=6,  # July default
        format_func=lambda x: {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }[x]
    )

    release_year = st.sidebar.number_input(
        "📆 Release Year", 
        min_value=2024, 
        max_value=2030, 
        value=2024
    )

    # Main content area
    col1, col2, col3 = st.columns([2, 2, 2])

    # Prediction button
    if st.sidebar.button("🚀 Predict Revenue", type="primary"):
        with st.spinner("Analyzing your movie..."):
            try:
                # Get prediction
                analysis = analyzer.analyze_roi(
                    budget=budget,
                    runtime=runtime, 
                    vote_average=vote_average,
                    release_month=release_month,
                    primary_genre=primary_genre,
                    vote_count=vote_count,
                    release_year=release_year
                )

                # Display key metrics
                with col1:
                    st.markdown("#### 💰 Financial Projections")
                    risk_class = {
                        'Low': 'success-metric',
                        'Medium': 'warning-metric', 
                        'High': 'danger-metric'
                    }[analysis['risk_level']]

                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>Predicted Revenue</h3>
                        <h2>${analysis['predicted_revenue']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Investment", f"${analysis['investment']:,.0f}")
                    st.metric("Profit", f"${analysis['profit']:,.0f}")

                with col2:
                    st.markdown("#### 📊 Performance Metrics")
                    roi_color = "🟢" if analysis['roi_percentage'] > 50 else "🟡" if analysis['roi_percentage'] > 0 else "🔴"

                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>ROI {roi_color}</h3>
                        <h2>{analysis['roi_percentage']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Risk Level", analysis['risk_level'])
                    profit_margin = (analysis['profit'] / analysis['predicted_revenue']) * 100
                    st.metric("Profit Margin", f"{profit_margin:.1f}%")

                with col3:
                    st.markdown("#### 🎯 Movie Details")
                    st.info(f"""
                    **Genre:** {primary_genre}  
                    **Runtime:** {runtime} minutes  
                    **Release:** {release_month}/{release_year}  
                    **Expected Rating:** {vote_average}/10  
                    **Target Audience:** {vote_count:,} votes
                    """)

                # ROI Gauge Chart
                st.markdown("---")
                st.markdown("#### 🎯 ROI Performance Gauge")

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = analysis['roi_percentage'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Return on Investment (%)"},
                    delta = {'reference': 50},  # Industry average
                    gauge = {
                        'axis': {'range': [None, 200]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "lightgreen"},
                            {'range': [100, 200], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Release timing optimization
                st.markdown("---")
                st.markdown("#### 📅 Optimal Release Timing Analysis")

                with st.spinner("Optimizing release timing..."):
                    timing_analysis = analyzer.optimize_release_timing(
                        budget=budget, runtime=runtime, vote_average=vote_average,
                        primary_genre=primary_genre, vote_count=vote_count, release_year=release_year
                    )

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.success(f"""
                    **Optimal Release Month:** {timing_analysis['best_month']}  
                    **Maximum ROI:** {timing_analysis['best_roi']:.1f}%  
                    **Improvement:** +{timing_analysis['best_roi'] - analysis['roi_percentage']:.1f}%
                    """)

                with col2:
                    # Monthly ROI chart
                    months = list(timing_analysis['monthly_roi'].keys())
                    rois = list(timing_analysis['monthly_roi'].values())
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    fig = px.bar(
                        x=[month_names[m-1] for m in months], 
                        y=rois,
                        title="ROI by Release Month",
                        labels={'x': 'Month', 'y': 'ROI (%)'},
                        color=rois,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Scenario comparison
                st.markdown("---")
                st.markdown("#### 🔍 Scenario Comparison")

                scenarios = {
                    'Conservative': {'budget_mult': 0.7, 'rating_adj': -0.3, 'votes_mult': 0.8},
                    'Current': {'budget_mult': 1.0, 'rating_adj': 0, 'votes_mult': 1.0},
                    'Optimistic': {'budget_mult': 1.3, 'rating_adj': 0.3, 'votes_mult': 1.2}
                }

                scenario_results = []
                for scenario, params in scenarios.items():
                    scenario_analysis = analyzer.analyze_roi(
                        budget=int(budget * params['budget_mult']),
                        runtime=runtime,
                        vote_average=min(10, vote_average + params['rating_adj']),
                        release_month=timing_analysis['best_month'],
                        primary_genre=primary_genre,
                        vote_count=int(vote_count * params['votes_mult']),
                        release_year=release_year
                    )
                    scenario_results.append({
                        'Scenario': scenario,
                        'Budget': f"${int(budget * params['budget_mult']):,}",
                        'Predicted Revenue': f"${scenario_analysis['predicted_revenue']:,.0f}",
                        'ROI': f"{scenario_analysis['roi_percentage']:.1f}%",
                        'Risk Level': scenario_analysis['risk_level'],
                        'Profit': f"${scenario_analysis['profit']:,.0f}"
                    })

                scenario_df = pd.DataFrame(scenario_results)
                st.dataframe(scenario_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

    # Information section
    st.markdown("---")
    st.markdown("#### ℹ️ About This Tool")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🤖 ML Model**  
        - Ensemble of RF, XGBoost, LightGBM  
        - 88.4% R² accuracy  
        - Trained on 13K+ movies
        """)

    with col2:
        st.markdown("""
        **📊 Features Used**  
        - Budget & Runtime  
        - Genre & Release timing  
        - Expected ratings & votes  
        """)

    with col3:
        st.markdown("""
        **🎯 Use Cases**  
        - Investment decisions  
        - Release strategy  
        - Risk assessment
        """)

if __name__ == "__main__":
    main()
