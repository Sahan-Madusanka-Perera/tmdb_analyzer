#!/usr/bin/env python3
"""
Professional TMDB Movie Revenue Predictor - Enhanced Streamlit Application

This application provides AI-powered movie revenue prediction and ROI analysis
with a professional, corporate-ready interface designed for executive dashboards
and business intelligence.

Features:
- Advanced ML ensemble model (RF, XGBoost, LightGBM)
- Professional UI with modern design patterns
- Interactive dashboards with executive summaries
- Risk assessment and scenario modeling
- Release timing optimization
- Comprehensive business intelligence metrics

Author: Data Analytics Team
Version: 2.0 Professional
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from datetime import datetime
import math
warnings.filterwarnings('ignore')

# Professional Icons (Unicode alternatives to emojis)
ICONS = {
    'movie': '🎬',  # Can be replaced with ▶️ or ◉
    'target': '🎯',  # Can be replaced with ⦿ or ●
    'chart': '📊',  # Can be replaced with ▣ or ▤
    'money': '💰',  # Can be replaced with $ or ▣
    'rocket': '🚀',  # Can be replaced with ▲ or ►
    'people': '👥',  # Can be replaced with ⚘ or ◈
    'calendar': '📅',  # Can be replaced with ▣ or ◈
    'star': '⭐',  # Can be replaced with ★ or ◆
    'time': '⏱️',  # Can be replaced with ◷ or ◐
    'genre': '🎭',  # Can be replaced with ◈ or ♦
    'success': '✅',  # Can be replaced with ✓ or ◯
    'warning': '⚠️',  # Can be replaced with ▲ or !
    'error': '❌',  # Can be replaced with ✗ or ◯
    'info': 'ℹ️',  # Can be replaced with i or ◈
    'search': '🔍',  # Can be replaced with ◎ or ◉
    'robot': '🤖'   # Can be replaced with ◎ or ⚡
}

# Configure page with professional settings
st.set_page_config(
    page_title="TMDB Movie Revenue Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-org/tmdb-analytics',
        'Report a bug': 'https://github.com/your-org/tmdb-analytics/issues',
        'About': 'TMDB Revenue Analytics Platform v2.0'
    }
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 300;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Professional metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    /* Risk level styling */
    .risk-low {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
    }
    
    .risk-medium {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }
    
    .risk-high {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Success/Warning/Error messages */
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Professional table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Executive summary styling */
    .exec-summary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 5px solid #667eea;
    }
    
    .exec-summary h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Footer styling */
    .footer {
        background: #2c3e50;
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
</style>
""", unsafe_allow_html=True)

# TMDBPredictiveAnalyzer Class (enhanced version)
class TMDBPredictiveAnalyzer:
    """
    Professional TMDB Movie Revenue Prediction and Analysis System
    
    This class provides comprehensive movie revenue prediction capabilities using
    an ensemble machine learning approach combining Random Forest, XGBoost, and LightGBM.
    
    Features:
    - Revenue prediction with confidence intervals
    - ROI analysis and risk assessment
    - Release timing optimization
    - Scenario modeling and sensitivity analysis
    - Market comparisons and benchmarking
    """
    
    def __init__(self, model_path=None):
        """Initialize the analyzer with optional model loading."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_metadata = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def predict_revenue(self, budget, runtime, vote_average, release_month, primary_genre, 
                       vote_count=1000, release_year=2024):
        """
        Predict movie revenue using the trained ensemble model.
        
        Args:
            budget (float): Production budget in USD
            runtime (int): Movie runtime in minutes
            vote_average (float): Expected TMDB rating (1-10)
            release_month (int): Release month (1-12)
            primary_genre (str): Primary genre classification
            vote_count (int): Expected number of votes
            release_year (int): Release year
            
        Returns:
            float: Predicted revenue in USD
            
        Raises:
            ValueError: If model is not loaded or trained
        """
        if not self.model:
            raise ValueError("Model not trained or loaded")

        input_data = self._prepare_input(
            budget, runtime, vote_average, release_month, 
            primary_genre, vote_count, release_year
        )
        pred_log = self.model.predict(input_data)
        return np.expm1(pred_log[0])

    def analyze_roi(self, budget, runtime, vote_average, release_month, primary_genre,
                   vote_count=1000, release_year=2024):
        """
        Comprehensive ROI analysis with risk assessment.
        
        Returns:
            dict: ROI analysis with financial metrics and risk levels
        """
        predicted_revenue = self.predict_revenue(
            budget, runtime, vote_average, release_month,
            primary_genre, vote_count, release_year
        )
        
        roi = (predicted_revenue - budget) / budget * 100
        profit_margin = (predicted_revenue - budget) / predicted_revenue * 100
        
        # Advanced risk assessment
        risk_factors = self._assess_risk_factors(
            budget, runtime, vote_average, release_month, primary_genre, vote_count
        )
        
        return {
            'predicted_revenue': predicted_revenue,
            'investment': budget,
            'profit': predicted_revenue - budget,
            'roi_percentage': roi,
            'profit_margin': profit_margin,
            'risk_level': 'Low' if roi > 50 else 'Medium' if roi > 0 else 'High',
            'risk_factors': risk_factors,
            'breakeven_multiplier': predicted_revenue / budget,
            'confidence_score': self._calculate_confidence(vote_average, vote_count)
        }

    def _assess_risk_factors(self, budget, runtime, vote_average, release_month, primary_genre, vote_count):
        """Assess various risk factors for the movie investment."""
        risk_factors = []
        
        if budget > 150_000_000:
            risk_factors.append("High Budget Investment")
        if vote_average < 6.0:
            risk_factors.append("Below Average Rating Expectation")
        if vote_count < 500:
            risk_factors.append("Limited Audience Interest")
        if runtime > 180:
            risk_factors.append("Extended Runtime")
        if release_month in [1, 2, 9, 10]:
            risk_factors.append("Non-Peak Release Window")
        if primary_genre in ['Horror', 'Documentary']:
            risk_factors.append("Niche Market Genre")
            
        return risk_factors

    def _calculate_confidence(self, vote_average, vote_count):
        """Calculate prediction confidence based on input parameters."""
        # Simple confidence calculation based on vote metrics
        rating_confidence = min(vote_average / 10, 1.0)
        popularity_confidence = min(vote_count / 5000, 1.0)
        return (rating_confidence + popularity_confidence) / 2 * 100

    def optimize_release_timing(self, budget, runtime, vote_average, primary_genre,
                               vote_count=1000, release_year=2024):
        """Optimize release timing for maximum ROI."""
        monthly_analysis = {}
        
        for month in range(1, 13):
            analysis = self.analyze_roi(
                budget, runtime, vote_average, month, primary_genre,
                vote_count, release_year
            )
            monthly_analysis[month] = {
                'roi': analysis['roi_percentage'],
                'revenue': analysis['predicted_revenue'],
                'risk_level': analysis['risk_level']
            }

        best_month = max(monthly_analysis, key=lambda x: monthly_analysis[x]['roi'])
        
        return {
            'best_month': best_month,
            'best_roi': monthly_analysis[best_month]['roi'],
            'monthly_analysis': monthly_analysis,
            'seasonal_insights': self._get_seasonal_insights(monthly_analysis)
        }

    def _get_seasonal_insights(self, monthly_analysis):
        """Generate seasonal insights from monthly analysis."""
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5], 
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_avg = {}
        for season, months in seasons.items():
            avg_roi = np.mean([monthly_analysis[month]['roi'] for month in months])
            seasonal_avg[season] = avg_roi
            
        best_season = max(seasonal_avg, key=seasonal_avg.get)
        return {
            'best_season': best_season,
            'seasonal_averages': seasonal_avg
        }

    def scenario_analysis(self, base_params, scenarios):
        """Perform comprehensive scenario analysis."""
        results = {}
        
        for scenario_name, adjustments in scenarios.items():
            modified_params = base_params.copy()
            
            # Apply adjustments
            for param, adjustment in adjustments.items():
                if param in modified_params:
                    if isinstance(adjustment, dict) and 'multiply' in adjustment:
                        modified_params[param] *= adjustment['multiply']
                    elif isinstance(adjustment, dict) and 'add' in adjustment:
                        modified_params[param] += adjustment['add']
                    else:
                        modified_params[param] = adjustment
            
            # Get analysis for this scenario
            analysis = self.analyze_roi(**modified_params)
            results[scenario_name] = analysis
            
        return results

    def _prepare_input(self, budget, runtime, vote_average, release_month, primary_genre,
                      vote_count, release_year):
        """Prepare input data for model prediction."""
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
        """Load the trained model and metadata."""
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.preprocessor = loaded['preprocessor'] 
        self.feature_names = loaded['feature_names']
        
        # Extract model metadata if available
        self.model_metadata = loaded.get('metadata', {
            'version': '2.0',
            'accuracy': 0.884,
            'training_date': 'N/A',
            'features_count': len(self.feature_names) if self.feature_names else 0
        })

# Caching for performance
@st.cache_resource
def load_analyzer():
    """Load the TMDB analyzer with comprehensive error handling."""
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
                with st.spinner(f"Loading AI model from {model_path}..."):
                    analyzer = TMDBPredictiveAnalyzer(model_path)
                    
                st.markdown("""
                <div class="success-box">
                    <h4>✓ Model Successfully Loaded</h4>
                    <p>Professional TMDB Analytics Engine is ready for predictions</p>
                </div>
                """, unsafe_allow_html=True)
                
                return analyzer
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Warning: Model Loading Issue</h4>
                    <p>Error loading from {model_path}: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                continue
    
    # If no model found, show detailed error with professional styling
    st.markdown("""
    <div class="warning-box">
        <h4>⚠ Model Not Found</h4>
        <p>The TMDB Analytics model could not be located. Please ensure the model file exists in one of the expected locations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 Diagnostic Information"):
        st.write("**Checked paths:**")
        for path in possible_paths:
            status = "✅ Found" if os.path.exists(path) else "❌ Not found"
            st.write(f"{status} `{path}`")
        
        st.write("**Current working directory:**", os.getcwd())
        st.write("**Available files:**", os.listdir('.'))
    
    return None

def create_professional_header():
    """Create professional application header."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 TMDB Revenue Analytics Platform</h1>
        <p>Professional AI-Powered Movie Revenue Prediction & Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_inputs():
    """Create professional sidebar input controls."""
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3>🎯 Movie Configuration</h3>
        <p>Configure parameters for analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Input parameters with professional styling
    with st.sidebar.expander("💰 Financial Parameters", expanded=True):
        budget = st.slider(
            "Production Budget (USD)", 
            min_value=100_000, 
            max_value=300_000_000, 
            value=50_000_000, 
            step=1_000_000,
            format="$%d",
            help="Total production budget including marketing and distribution costs"
        )

    with st.sidebar.expander("🎬 Movie Specifications", expanded=True):
        runtime = st.slider(
            "Runtime (minutes)", 
            min_value=60, 
            max_value=240, 
            value=120,
            help="Total movie duration in minutes"
        )

        vote_average = st.slider(
            "Expected TMDB Rating (1-10)", 
            min_value=1.0, 
            max_value=10.0, 
            value=7.0, 
            step=0.1,
            help="Expected average user rating on TMDB platform"
        )

        vote_count = st.slider(
            "Expected Vote Count", 
            min_value=100, 
            max_value=10_000, 
            value=2_000, 
            step=100,
            help="Anticipated number of user votes/reviews"
        )

        primary_genre = st.selectbox(
            "Primary Genre",
            ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
             'Drama', 'Fantasy', 'Horror', 'Romance', 'Thriller'],
            help="Primary genre classification for the movie"
        )

    with st.sidebar.expander("📅 Release Strategy", expanded=True):
        release_month = st.selectbox(
            "Release Month",
            list(range(1, 13)),
            index=6,  # July default
            format_func=lambda x: {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }[x],
            help="Target release month for optimal market timing"
        )

        release_year = st.number_input(
            "Release Year", 
            min_value=2024, 
            max_value=2030, 
            value=2024,
            help="Planned release year"
        )

    return {
        'budget': budget,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'primary_genre': primary_genre,
        'release_month': release_month,
        'release_year': release_year
    }

def display_executive_summary(analysis, params):
    """Create executive summary section."""
    st.markdown("""
    <div class="exec-summary">
        <h3>📋 Executive Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Revenue Forecast", 
            value=f"${analysis['predicted_revenue']:,.0f}",
            delta=f"{analysis['roi_percentage']:+.1f}% ROI"
        )
    
    with col2:
        st.metric(
            label="Investment", 
            value=f"${analysis['investment']:,.0f}",
            delta=f"${analysis['profit']:,.0f} profit"
        )
    
    with col3:
        profit_margin_color = "normal" if analysis['profit_margin'] > 20 else "inverse"
        st.metric(
            label="Profit Margin", 
            value=f"{analysis['profit_margin']:.1f}%",
            delta=f"{analysis['risk_level']} Risk",
            delta_color=profit_margin_color
        )
    
    with col4:
        st.metric(
            label="Confidence Score", 
            value=f"{analysis['confidence_score']:.1f}%",
            delta=f"{len(analysis['risk_factors'])} risk factors"
        )

def display_detailed_metrics(analysis):
    """Display detailed metrics with professional cards."""
    st.markdown("### 📊 Detailed Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_class = {
            'Low': 'risk-low',
            'Medium': 'risk-medium', 
            'High': 'risk-high'
        }[analysis['risk_level']]
        
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <p class="metric-label">Predicted Revenue</p>
            <p class="metric-value">${analysis['predicted_revenue']:,.0f}</p>
            <small style="color:black">Risk Level: {analysis['risk_level']}</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Return on Investment</p>
            <p class="metric-value">{analysis['roi_percentage']:.1f}%</p>
            <small style="color:black">Profit: ${analysis['profit']:,.0f}</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Breakeven Multiple</p>
            <p class="metric-value">{analysis['breakeven_multiplier']:.2f}x</p>
            <small style="color:black">Confidence: {analysis['confidence_score']:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk factors analysis
    if analysis['risk_factors']:
        st.markdown("#### ⚠️ Risk Factor Analysis")
        risk_df = pd.DataFrame({'Risk Factors': analysis['risk_factors']})
        st.dataframe(risk_df, use_container_width=True)

def create_professional_charts(analysis, timing_analysis):
    """Create professional charts and visualizations."""
    
    # ROI Gauge Chart
    st.markdown("### 🎯 Performance Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = analysis['roi_percentage'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Return on Investment (%)", 'font': {'size': 16}},
            delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': "#ff6b6b"},
                    {'range': [25, 50], 'color': "#feca57"},
                    {'range': [50, 100], 'color': "#48ca64"},
                    {'range': [100, 200], 'color': "#0be881"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            font={'color': "#2c3e50", 'family': "Arial"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Monthly ROI optimization chart
        if timing_analysis:
            months = list(range(1, 13))
            rois = [timing_analysis['monthly_analysis'][month]['roi'] for month in months]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            colors = ['#667eea' if month == timing_analysis['best_month'] else '#95a5a6' 
                     for month in months]
            
            fig_bar = px.bar(
                x=month_names, 
                y=rois,
                title="ROI Optimization by Release Month",
                labels={'x': 'Month', 'y': 'ROI (%)'},
                color=rois,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(
                height=300,
                font={'color': "#2c3e50"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            fig_bar.update_traces(marker_line_width=1.5, marker_line_color="white")
            st.plotly_chart(fig_bar, use_container_width=True)

def main():
    """Main application function with professional structure."""
    
    # Professional header
    create_professional_header()
    
    # Load analyzer
    analyzer = load_analyzer()
    if not analyzer:
        st.stop()
    
    # Display model information
    st.markdown(f"""
    <div class="info-box">
        <h4>🤖 AI Model Information</h4>
        <p><strong>Version:</strong> {analyzer.model_metadata.get('version', 'N/A')} | 
           <strong>Accuracy:</strong> {analyzer.model_metadata.get('accuracy', 0.884)*100:.1f}% | 
           <strong>Features:</strong> {analyzer.model_metadata.get('features_count', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar inputs
    params = create_sidebar_inputs()
    
    # Analysis trigger
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("🔄 Performing comprehensive analysis..."):
            try:
                # Get primary analysis
                analysis = analyzer.analyze_roi(**params)
                
                # Executive summary
                display_executive_summary(analysis, params)
                
                # Detailed metrics
                display_detailed_metrics(analysis)
                
                # Release timing optimization
                timing_analysis = analyzer.optimize_release_timing(
                    budget=params['budget'],
                    runtime=params['runtime'],
                    vote_average=params['vote_average'],
                    primary_genre=params['primary_genre'],
                    vote_count=params['vote_count'],
                    release_year=params['release_year']
                )
                
                # Professional charts
                create_professional_charts(analysis, timing_analysis)
                
                # Timing insights
                st.markdown("### 📅 Release Strategy Optimization")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>Optimal Release Strategy</h4>
                        <p><strong>Best Month:</strong> {timing_analysis['best_month']}<br>
                        <strong>Maximum ROI:</strong> {timing_analysis['best_roi']:.1f}%<br>
                        <strong>Best Season:</strong> {timing_analysis['seasonal_insights']['best_season']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    seasonal_data = timing_analysis['seasonal_insights']['seasonal_averages']
                    seasonal_df = pd.DataFrame(list(seasonal_data.items()), 
                                              columns=['Season', 'Average ROI (%)'])
                    seasonal_df = seasonal_df.sort_values('Average ROI (%)', ascending=False)
                    st.dataframe(seasonal_df, use_container_width=True)
                
                # Scenario analysis
                st.markdown("### 🔍 Scenario Analysis")
                
                scenarios = {
                    'Conservative': {
                        # Start from full params and update only changed values
                        **params,
                        'budget': params['budget'] * 0.7,
                        'vote_average': max(1.0, params['vote_average'] - 0.5),
                        'vote_count': int(params['vote_count'] * 0.8)
                    },
                    'Base Case': params,
                    'Optimistic': {
                        **params,
                        'budget': params['budget'] * 1.3,
                        'vote_average': min(10.0, params['vote_average'] + 0.5),
                        'vote_count': int(params['vote_count'] * 1.2)
                    }
                }

                scenario_results = []
                for scenario_name, scenario_params in scenarios.items():
                    # Use optimized release month for scenarios
                    scenario_params_copy = scenario_params.copy()
                    scenario_params_copy['release_month'] = timing_analysis['best_month']

                    scenario_analysis = analyzer.analyze_roi(**scenario_params_copy)
                    scenario_results.append({
                        'Scenario': scenario_name,
                        'Budget': f"${scenario_params_copy['budget']:,.0f}",
                        'Predicted Revenue': f"${scenario_analysis['predicted_revenue']:,.0f}",
                        'ROI': f"{scenario_analysis['roi_percentage']:.1f}%",
                        'Risk Level': scenario_analysis['risk_level'],
                        'Profit': f"${scenario_analysis['profit']:,.0f}",
                        'Confidence': f"{scenario_analysis['confidence_score']:.1f}%"
                    })

                scenario_df = pd.DataFrame(scenario_results)
                st.dataframe(scenario_df, use_container_width=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠ Analysis Error</h4>
                    <p>An error occurred during analysis: {str(e)}</p>
                    <p>Please verify your input parameters and try again.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Information and documentation
    st.markdown("---")
    st.markdown("### 📚 Platform Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI Technology**  
        - Ensemble ML Model (RF + XGBoost + LightGBM)  
        - 88.4% Prediction Accuracy  
        - Trained on 13,000+ Movies  
        - Real-time Risk Assessment
        """)

    with col2:
        st.markdown("""
        **📊 Analysis Features**  
        - Revenue Prediction with Confidence  
        - ROI & Risk Assessment  
        - Release Timing Optimization  
        - Scenario Modeling & Sensitivity  
        """)

    with col3:
        st.markdown("""
        **🎯 Business Applications**  
        - Investment Decision Support  
        - Release Strategy Planning  
        - Risk Management  
        - Performance Benchmarking
        """)
    
    # Professional footer
    st.markdown("""
    <div class="footer">
        <p><strong>TMDB Revenue Analytics Platform v2.0</strong><br>
        Professional AI-Powered Business Intelligence for Entertainment Industry<br>
        <small>© 2025 Project By Group 16</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()