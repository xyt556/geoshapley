# GeoShapley Interactive Analysis Tool - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Introduction

GeoShapley Interactive Analysis Tool is a web-based application built with Streamlit that provides an intuitive interface for analyzing spatial effects in machine learning models using the GeoShapley method. This tool allows users to:

- Upload and prepare spatial data
- Train machine learning models
- Perform GeoShapley analysis
- Visualize spatial effects and feature contributions
- Calculate spatially varying coefficients (SVC)

### What is GeoShapley?

GeoShapley is a game theory-based approach to measuring spatial effects from machine learning models. It decomposes model predictions into:
- **Primary effects**: Non-spatial feature contributions
- **Spatial effects**: Location-specific contributions
- **Spatial interaction effects**: Interactions between features and location

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install basic requirements
pip install -r requirements.txt

# Install additional packages for full functionality
pip install streamlit
pip install "flaml[automl]"
pip install geopandas matplotlib seaborn jupyter

# Optional: For GWR smoothing in SVC calculation
pip install mgwr

# Optional: For GAM smoothing curves
pip install pygam
```

### Step 3: Install GeoShapley Package

```bash
pip install -e .
```

---

## Quick Start

### 1. Launch the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### 2. Basic Workflow

1. **Data Upload** → Upload CSV file or use example data
2. **Model Training** → Train a machine learning model
3. **GeoShapley Analysis** → Run spatial effect analysis
4. **Results Visualization** → View and download results

---

## Detailed Usage

### Page 1: Data Upload

#### Upload Your Data

1. Click **"Upload CSV File"** button
2. Select a CSV file containing:
   - Feature columns (predictors)
   - Target variable (what you want to predict)
   - Spatial coordinate columns (e.g., longitude/latitude or UTM_X/UTM_Y)

**OR**

Use the example data by checking **"Use Example Data (Seattle House Price Data)"**

#### Configure Your Data

1. **Select Target Variable (y)**: Choose the column you want to predict
2. **Select Task Type**: 
   - **Regression**: For continuous values (e.g., house prices, temperature)
   - **Classification**: For categorical values (e.g., yes/no, categories)
3. **Select Feature Columns**: Choose all predictor columns (excluding spatial coordinates)
4. **Select Spatial Coordinate Columns**: Choose coordinate columns (must be placed last)

**Important**: Spatial coordinate columns must be the last columns in your dataset!

5. Click **"Prepare Data"** button

#### Data Format Requirements

- CSV file format
- Spatial coordinates must be in the last columns
- No missing values in coordinate columns
- Numeric data types for features and coordinates

---

### Page 2: Model Training

#### View Data Information

The page displays:
- Number of samples
- Number of features
- Number of spatial coordinates

#### Select Model Type

**AutoML (FLAML)** (Recommended)
- Automatically selects the best model
- Supports multiple algorithms (XGBoost, LightGBM, Random Forest, etc.)
- Set time budget (seconds) for training

**Random Forest**
- Classic ensemble method
- Set number of trees (10-500)
- Good for most problems

**Neural Network (MLP)**
- Multi-layer perceptron
- Set hidden layer structure (e.g., "100,50" for two hidden layers)
- Good for complex non-linear relationships

#### Configure Training Parameters

- **Test Set Ratio**: Proportion of data for testing (default: 0.2)
- **Random Seed**: For reproducibility (default: 42)
- **Model-specific parameters**: As shown above

#### Train Model

1. Click **"Start Training"** button
2. Wait for training to complete
3. View performance metrics:
   - **R² Score** (for regression) or **Accuracy** (for classification)
   - **MSE** (Mean Squared Error for regression)

---

### Page 3: GeoShapley Analysis

#### Configure Analysis Parameters

**Background Data Method**
- **K-means Clustering**: Recommended, uses representative samples
- **Random Sampling**: Random selection of background samples

**Background Data Size**: Number of samples used as background (default: 100)
- Larger values = more accurate but slower
- Recommended: 50-200 samples

**Number of Parallel Jobs**: 
- -1 = Use all CPU cores (recommended)
- Positive number = Use specific number of cores

**Number of Spatial Coordinates (g)**:
- Usually 2 (longitude, latitude)
- Can be more for higher-dimensional spatial data

**Analysis Sample Size**: 
- Number of samples to analyze (default: 100)
- Larger values = more comprehensive but slower
- Start with 100, increase if needed

#### Run Analysis

1. Click **"Start Analysis"** button
2. Wait for analysis to complete (may take several minutes)
3. View results:
   - **Basic Statistics**: Summary statistics for all GeoShapley values
   - **Additivity Check**: Verifies that components add up to model predictions

---

### Page 4: Results Visualization

#### Visualization Options

**1. Summary Plot**
- SHAP-style summary plot showing feature importance
- Options:
  - Include interaction effects (recommended)
  - Image resolution (DPI)
- Click **"Generate Summary Plot"**

**2. Partial Dependence Plot**
- Shows how each feature affects predictions
- Options:
  - Max columns per row
  - Show GAM smoothing curve (requires pygam)
- Click **"Generate Partial Dependence Plot"**

**3. Contribution Bar Plot**
- Ranked bar chart of feature contributions
- Shows non-spatial vs spatial contributions
- Click **"Generate Contribution Bar Plot"**

**4. Summary Statistics Table**
- Detailed statistics table
- Options:
  - Include interaction effects
- Download as CSV

#### Spatially Varying Coefficients (SVC)

**Purpose**: Calculate location-specific coefficients for features

**Configuration**:
1. **Select Features**: Choose features to calculate SVC for
2. **Coefficient Type**:
   - **raw**: Raw coefficients (may contain extreme values)
   - **gwr**: GWR-smoothed coefficients (requires mgwr package, recommended)
3. **Include Primary Effects**: Whether to include primary effects in coefficients

**Visualization Options**:
- **Scatter Plot**: Shows SVC values in coordinate space
- **Heatmap**: Matrix visualization of SVC values
- **Statistical Plot**: Box plots showing SVC distributions

**Download**: SVC data can be downloaded as CSV

---

## Features

### Key Features

1. **User-Friendly Interface**: No coding required, all operations through GUI
2. **Multiple Model Support**: AutoML, Random Forest, Neural Networks
3. **Comprehensive Analysis**: Full GeoShapley decomposition
4. **Rich Visualizations**: Multiple plot types for different insights
5. **Export Capabilities**: Download results as CSV files
6. **Example Data**: Built-in Seattle house price dataset for testing

### Advanced Features

- **Parallel Processing**: Utilizes multiple CPU cores for faster analysis
- **Background Data Optimization**: K-means clustering for efficient background selection
- **Spatial Smoothing**: GWR-based smoothing for SVC calculation
- **Additivity Verification**: Automatic check that components sum correctly

---

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
```

#### 2. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'geoshapley'`

**Solution**:
```bash
pip install -e .
```

#### 3. AutoML Training Fails

**Problem**: AutoML shows warnings or errors

**Solution**:
```bash
pip install "flaml[automl]"
```

#### 4. Summary Plot is Empty

**Problem**: Plot appears but shows no data

**Solution**: 
- Ensure GeoShapley analysis completed successfully
- Check that data was properly prepared
- Try regenerating the plot

#### 5. SVC Calculation Fails

**Problem**: Error when calculating SVC with GWR method

**Solution**:
```bash
pip install mgwr
```

#### 6. GAM Smoothing Not Available

**Problem**: GAM curve option doesn't work

**Solution**:
```bash
pip install pygam
```

#### 7. Memory Errors

**Problem**: Out of memory errors with large datasets

**Solutions**:
- Reduce analysis sample size
- Reduce background data size
- Use fewer features
- Close other applications

#### 8. Slow Performance

**Problem**: Analysis takes too long

**Solutions**:
- Reduce sample size
- Use fewer features
- Reduce background data size
- Use parallel processing (set n_jobs=-1)
- Use faster model (Random Forest instead of Neural Network)

---

## FAQ

### Q1: What data format is required?

**A**: CSV format with:
- Feature columns (numeric)
- Target variable column
- Spatial coordinate columns (must be last columns)
- No missing values in coordinates

### Q2: How many spatial coordinates can I use?

**A**: Typically 2 (longitude/latitude), but the tool supports up to 5. Set the `g` parameter accordingly.

### Q3: What's the difference between raw and GWR coefficients?

**A**: 
- **Raw**: Direct calculation, may have extreme values
- **GWR**: Geographically Weighted Regression smoothing, produces smoother, more interpretable results

### Q4: How long does analysis take?

**A**: Depends on:
- Sample size (most important)
- Number of features
- Background data size
- Model complexity
- Typically 1-10 minutes for 100 samples

### Q5: Can I use my own trained model?

**A**: Currently, the tool trains models internally. For custom models, you would need to modify the code to load pre-trained models.

### Q6: What models are supported?

**A**: 
- AutoML (FLAML) - automatically selects best
- Random Forest
- Neural Network (MLP)
- Any model compatible with scikit-learn interface

### Q7: Can I analyze classification problems?

**A**: Yes! Select "Classification" as task type when preparing data.

### Q8: What do the GeoShapley values mean?

**A**: 
- **Primary effects**: How features affect predictions regardless of location
- **Spatial effects**: How location itself affects predictions
- **Interaction effects**: How feature effects vary by location

### Q9: How do I interpret the summary plot?

**A**: 
- Features are ranked by importance (top to bottom)
- Color indicates feature value (red = high, blue = low)
- Horizontal position shows impact on prediction
- Wider spread = more variable effect

### Q10: Can I save my analysis results?

**A**: Yes! You can download:
- Summary statistics as CSV
- SVC coefficients as CSV
- Visualizations can be saved using browser's save image feature

---

## Tips for Best Results

1. **Start Small**: Begin with 50-100 samples to test the workflow
2. **Use Example Data**: Try the Seattle house price example first
3. **Check Data Quality**: Ensure coordinates are valid and features are numeric
4. **Use K-means**: K-means clustering usually works better than random sampling
5. **Parallel Processing**: Always use n_jobs=-1 for faster analysis
6. **AutoML First**: Try AutoML before specific models - it often finds better solutions
7. **Interpret Results**: Review additivity check to ensure analysis is correct
8. **Export Results**: Download important results for further analysis

---

## Support and Resources

### Documentation
- GeoShapley GitHub: https://github.com/Ziqi-Li/geoshapley
- GeoShapley Paper: Li, Z. (2024). GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models

### Related Tools
- SHAP: https://github.com/slundberg/shap
- FLAML: https://github.com/microsoft/FLAML
- Streamlit: https://streamlit.io/

---

## Version History

- **v1.0** (Current): Initial release with full GeoShapley analysis capabilities

---

## License

This tool is provided as-is for research and educational purposes. Please refer to the GeoShapley project license for details.

---

## Contact

For issues related to:
- **GeoShapley library**: See GitHub repository
- **This application**: Check troubleshooting section or modify code as needed

---

**Happy Analyzing! 🌍📊**

