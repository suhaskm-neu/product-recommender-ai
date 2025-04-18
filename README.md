# Product Recommender AI

## Introduction

Product Recommender AI is an MLOps-driven engine that fuses advanced machine learning with deep user and social network insights for personalized, real-time product suggestions. This project explores various approaches to solve the challenging problem of next-item prediction in e-commerce and content recommendation systems.

### Problem Statement

In modern digital platforms, predicting the next item a user will interact with is crucial for enhancing user experience and increasing engagement. This project addresses several key challenges:

- **Extreme Data Sparsity**: Handling datasets with thousands of unique items where most items appear only once
- **Memory Efficiency**: Developing models that can scale to large item catalogs (50,000+ items)
- **Cold Start Problem**: Providing recommendations for new users with limited interaction history
- **Real-time Personalization**: Balancing accuracy with computational efficiency for timely recommendations

### Project Goals

The primary objectives of this project are to:

1. Develop and compare different recommendation approaches, from traditional machine learning to specialized Markov models
2. Identify the most effective techniques for sparse, large-scale recommendation scenarios
3. Implement memory-efficient algorithms suitable for production environments
4. Leverage social network insights to enhance recommendation quality
5. Provide a comprehensive framework for next-item prediction that can be extended and customized

### Key Features

- **Multiple Model Implementations**: Random Forest, Markov Chain models, Focused models, and Social Network-enhanced recommendations
- **Progressive Model Improvement**: Documented evolution from basic models to sophisticated approaches
- **Comprehensive Evaluation**: Various metrics including accuracy, Top-N recommendations, and performance analysis
- **Memory-Optimized Algorithms**: Techniques specifically designed for large item catalogs
- **Social Network Integration**: Leveraging user relationships to enhance recommendation quality

This project demonstrates how different approaches perform under various conditions, with a focus on practical, deployable solutions for real-world recommendation systems.


## Project Architecture

### System Components

The Product Recommender AI system consists of several key components:

1. **Data Processing Pipeline**: Handles data ingestion, cleaning, and transformation
   - Processes user-item interactions from different data formats
   - Creates feature representations suitable for model training
   - Generates training and testing datasets

2. **Model Framework**: Contains multiple recommendation approaches
   - **Traditional ML Models**: Random Forest implementation for baseline comparison
   - **Markov Chain Models**: Various implementations from basic to focused models
   - **Social Network Models**: Leverages user relationships for enhanced recommendations
   - **Hybrid Approaches**: Combines multiple techniques for improved performance

3. **Evaluation System**: Comprehensive assessment of model performance
   - Accuracy metrics for exact match predictions
   - Top-N recommendation evaluation
   - Performance analysis across different data characteristics

4. **Persistence Layer**: Stores trained models and prediction results
   - Model serialization for future use
   - Results logging and tracking

### Data Flow

The system follows this general data flow:

1. Raw user-item interaction data is ingested from CSV files
2. Data is preprocessed to handle different timestamp formats and create next-item targets
3. For traditional ML models, features are extracted and scaled
4. For Markov models, transition matrices are built from sequential interactions
5. Models are trained on historical data
6. Predictions are generated for test sequences
7. Results are evaluated and compared across models
8. The best-performing models are saved for future use

### Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical operations and array handling
- **Scikit-learn**: Implementation of traditional ML models and evaluation metrics
- **MLflow**: Experiment tracking and model management
- **Matplotlib/Seaborn**: Visualization of results and data characteristics
- **NetworkX**: Social network analysis and graph-based recommendations

## Data Description

### Dataset Formats

The project utilizes two primary data formats:

1. **Main Dataset (`user_item_interactions.csv`)**
   - Contains user-item interaction records
   - Timestamps in datetime string format (e.g., '2025-02-17 22:52:36.850723')
   - Includes user_id, item_id, timestamp, view_time, and click_rate

2. **Sample Dataset (`user_0_processed.csv`)**
   - Focused on a single user's interactions
   - Timestamps as Unix timestamps (float values like 1739832700.0)
   - Includes an additional 'next_item_id' column for training

### Data Characteristics

Analysis of the datasets revealed several important characteristics:

1. **Extreme Sparsity**
   - 50,511 unique items in 80,000 records
   - Most common item appears only 8 times
   - Average item appears just 1.58 times
   - Median item appears exactly once

2. **Limited Pattern Repetition**
   - Only 0.00% of items have any repeated transitions
   - Only 0.00% of sequences appear more than once
   - Total repeated transitions: 1 out of 79,999 (0.00%)

3. **Session Structure**
   - Data appears as a single continuous session
   - No natural breaks in user activity
   - Sequential nature of interactions is critical for modeling

### Data Challenges

These characteristics present significant challenges for recommendation systems:

1. **Cold Start Problem**: With most items appearing only once, traditional collaborative filtering approaches struggle
2. **Memory Constraints**: The large number of unique items (50,000+) creates memory issues for many model types
3. **Sparse Transition Matrices**: Item-to-item transition matrices are extremely sparse, making pattern identification difficult
4. **Evaluation Complexity**: The lack of repeated patterns makes traditional evaluation metrics less meaningful

### Preprocessing Steps

To address these challenges, several preprocessing steps were implemented:

1. **Data Cleaning**
   - Handling different timestamp formats
   - Removing duplicate entries
   - Sorting by user and timestamp

2. **Feature Engineering**
   - Creating 'next_item_id' target by shifting item sequences
   - Calculating additional features like view_time and click_rate
   - Encoding categorical features

3. **Data Filtering**
   - Focusing on the most frequent items for some models
   - Creating balanced training and testing splits
   - Implementing various sampling strategies for model evaluation

## Model Evolution

This project demonstrates a progressive journey through various recommendation approaches, each addressing specific challenges and building upon previous insights.

### 1. Traditional Machine Learning Approach

The initial approach utilized a Random Forest classifier:

- **Implementation**: `train_baseline.py`
- **Features Used**: 
  - item_id
  - prev_item_id
  - view_time
  - click_rate
  - Previous interaction patterns
- **Challenges Addressed**:
  - Memory constraints by focusing on top 100 most frequent items
  - Feature scaling for improved performance
- **Results**:
  - 0% accuracy on test data
  - Revealed fundamental limitations of traditional ML for this sparse dataset

### 2. Basic Markov Chain Model

The next evolution introduced a simple Markov model:

- **Implementation**: `minimal_model.py`
- **Approach**:
  - Created item-to-item transition matrices
  - For each item, predicted the most common next item based on historical data
- **Advantages**:
  - Memory-efficient representation
  - Simple yet effective for sequential data
  - No feature engineering required
- **Results**:
  - ~23% accuracy on filtered data
  - Significantly outperformed the traditional ML approach

### 3. Focused Markov Model

Building on the success of the basic Markov model:

- **Implementation**: `focused_markov_model.py`
- **Enhancements**:
  - Strategic focus on most frequent items
  - Improved transition probability calculations
  - Optimized memory usage
- **Evaluation Approaches**:
  - Exact match accuracy
  - Top-N recommendation metrics
- **Results**:
  - Full Model on Full Data: 40.29% accuracy
  - Top-1: 61.82% accuracy
  - Top-3 and Top-5: 100% accuracy

### 4. Advanced Markov Variants

Several advanced Markov model variants were explored:

- **Higher-Order Markov Models**:
  - Considered multiple previous items for prediction
  - Addressed sequence dependencies beyond immediate transitions
  
- **Time-Decay Models**:
  - Weighted recent interactions more heavily
  - Incorporated temporal dynamics into predictions
  
- **Hybrid and Fallback Models**:
  - Combined multiple approaches
  - Implemented fallback strategies for unseen patterns

### 5. Social Network Enhanced Models

The final evolution incorporated social network insights:

- **Implementation**: `social_recommendation.py`
- **Approach**:
  - Generated synthetic social connections between users
  - Enhanced recommendations using collaborative information
  - Balanced individual preferences with social influences
- **Advantages**:
  - Addressed cold start problem for new users
  - Leveraged collective intelligence
  - Improved recommendation diversity
- **Results**:
  - Enhanced performance for users with limited history
  - Provided more robust recommendations across different user segments

Each model iteration provided valuable insights into the challenges of next-item prediction in sparse datasets, ultimately leading to a comprehensive recommendation framework that balances accuracy, memory efficiency, and practical applicability.

## Key Findings and Results

Throughout the development of this recommendation system, several key findings emerged that provide valuable insights for building effective next-item prediction models.

### Performance Comparison

| Model Type | Accuracy (%) | Memory Usage | Training Time | Prediction Time |
|------------|--------------|--------------|---------------|-----------------|
| Random Forest (Traditional ML) | 0.00 | High | Long | Medium |
| Basic Markov Model | 23.42 | Low | Fast | Very Fast |
| Focused Markov Model (Full Data) | 40.29 | Medium | Medium | Fast |
| Focused Markov Model (Top-1) | 61.82 | Low | Fast | Very Fast |
| Focused Markov Model (Top-3) | 100.00 | Low | Fast | Very Fast |
| Social Network Enhanced | Varies | Medium | Medium | Medium |

### Critical Insights

1. **Data Sparsity Impact**
   - Traditional ML models fail completely on extremely sparse data
   - Item-based approaches outperform user-based approaches in sparse environments
   - Even simple Markov models can capture meaningful patterns when properly implemented

2. **Evaluation Methodology Matters**
   - Top-N evaluation provides a more realistic assessment than exact match accuracy
   - The choice of test set significantly impacts reported performance
   - Focusing evaluation on popular items yields more meaningful results

3. **Memory-Efficiency Tradeoffs**
   - Focusing on the most frequent items (top 100) dramatically improves both performance and memory usage
   - Full models trained on all data but evaluated on filtered subsets outperform models both trained and evaluated on filtered data
   - Strategic data filtering preserves 99.31% of memory while maintaining high accuracy

4. **Model Complexity vs. Performance**
   - Simpler models often outperform complex ones in extremely sparse environments
   - Higher-order Markov models show diminishing returns due to the lack of repeated patterns
   - Hybrid approaches combining multiple strategies provide the most robust performance

### Data-Driven Discoveries

Our detailed analysis of the user_0_processed.csv file revealed why recommendation models struggle with this dataset:

1. **Extreme Repetition Scarcity**
   - Only 1 out of 50,511 items (0.00%) has any repeated transitions
   - Only 1 out of 79,997 sequences (0.00%) appears more than once
   - Total repeated transitions: 1 out of 79,999 (0.00%)

2. **Session Characteristics**
   - The entire dataset appears to be a single continuous session
   - No natural breaks in user activity make session-based approaches challenging
   - The sequential nature of the data is critical for any successful model

### Practical Applications

These findings translate into practical recommendations for real-world recommendation systems:

1. **Focus on Meaningful Patterns**
   - Concentrate computational resources on the most frequent items and transitions
   - Implement fallback strategies for rare or unseen items
   - Balance between personalization and generalization

2. **Use Balanced Evaluation**
   - Employ Top-N metrics rather than exact match accuracy
   - Consider business objectives when defining success metrics
   - Test models on realistic subsets of data

3. **Implement Realistic Recommendation Scenarios**
   - Present users with multiple options (Top-3 or Top-5) rather than a single prediction
   - Incorporate social and contextual signals when available
   - Design systems that can adapt to different levels of data sparsity

4. **Leverage Full Context**
   - Train models on all available data when possible
   - Focus evaluation on the most relevant subset of predictions
   - Combine multiple approaches for robust performance across different scenarios


## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Getting Started

1. **Clone the repository** (or download the ZIP file)
   ```bash
   git clone [https://github.com/yourusername/product-recommender-ai.git](https://github.com/yourusername/product-recommender-ai.git)
   cd product-recommender-ai

2. **Create a virtual environment** (optional but recommended)
   ```bash
   # Using venv
   python -m venv venv

   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Project Structure

```
product-recommender-ai/
├── README.md
├── requirements.txt
├── data/
│   ├── samples/
│   │   ├── user_0_processed.csv
│   │   ├── user_1_processed.csv
│   │   └── ...
│   └── full_data.csv
├── src/
│   ├── EDA_on_the_go.ipynb
│   ├── markov_chains/
│   │   ├── markov_single_user.py
│   │   ├── markov_multi_user.py
│   │   └── ...
│   └── ...
└── ...
```

### Configuration

Most models can be configured by modifying parameters at the top of their respective Python files. Common configuration options include:

- `top_n`: Number of top items to recommend
- `TEST_SIZE`: Proportion of data to use for testing (default: 0.2)
- `RANDOM_STATE`: Seed for random operations (default: 42)
- `ORDER`: Order of Markov model for higher-order implementations (default: 1)

### Data Requirements

### Data Requirements

To run the models with your own data, ensure your dataset follows one of these formats:

| Format | Required Columns | Description |
|--------|------------------|-------------|
| Format 1 | `user_id`, `item_id`, `timestamp`, `view_time`, `click_rate` | Standard interaction data |
| Format 2 | `user_id`, `item_id`, `timestamp`, `next_item_id` | Pre-processed data with next item |

The timestamps can be either datetime strings (e.g., '2025-02-17 22:52:36') or Unix timestamps (float values).


## Usage Guide

### Running the Models

#### 1. Random Forest Baseline Model

```bash
python src/train_baseline.py
```

This script:

- Loads and preprocesses the dataset  
- Trains a Random Forest model on the top 100 most frequent items  
- Evaluates performance on a test set  
- Saves the trained model to the `models` directory  

#### 2. Basic Markov Model

```bash
python src/minimal_model.py
```

This script:

- Implements a simple first-order Markov model  
- Creates an item-to-item transition matrix  
- Predicts the most likely next item for each item in the test set  
- Reports accuracy and saves the model  

#### 3. Focused Markov Model

```bash
python src/markov_chains/focused_markov_model.py
```

This script:

- Implements an enhanced Markov model focusing on frequent items  
- Provides multiple evaluation metrics including Top-N accuracy  
- Generates detailed performance reports and visualizations  

#### 4. Multi-User Markov Model

```bash
python src/markov_chains/markov_multi_user.py
```

This script:

- Extends the Markov approach to handle multiple users  
- Creates user-specific transition matrices  
- Combines individual and collective patterns for prediction  

#### 5. Social Network Enhanced Model

```bash
python src/social_networks/social_recommendation.py
```

This script:

- Generates synthetic social connections between users  
- Enhances recommendations using social network information  
- Evaluates the impact of social signals on recommendation quality  

### Example: Training and Evaluating a Focused Markov Model

```python
# Import the model
from src.markov_chains.focused_markov_model import FocusedMarkovModel

# Initialize the model
model = FocusedMarkovModel(top_n=100)

# Load and preprocess data
model.load_data('data/user_item_interactions.csv')

# Train the model
model.train()

# Evaluate on test data
accuracy = model.evaluate()
print(f"Model accuracy: {accuracy:.2f}%")

# Get Top-N recommendations for a specific item
recommendations = model.predict(item_id=12345, n=5)
print("Top 5 recommendations:", recommendations)

# Save the model
model.save('models/focused_markov_model.pkl')
```

### Visualizing Results

Many of the model implementations include built-in visualization capabilities:

```python
# After training and evaluating a model
model.plot_accuracy_comparison()  # Compare with baseline models
model.plot_confusion_matrix()     # Visualize prediction patterns
model.plot_top_n_accuracy()       # Show Top-N performance
```

### Batch Prediction

For batch prediction on new data:

```python
# Load a saved model
import pickle
with open('models/focused_markov_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new data
import pandas as pd
new_data = pd.read_csv('data/new_interactions.csv')

# Generate predictions
predictions = model.batch_predict(new_data)

# Save predictions
predictions.to_csv('predictions.csv', index=False)
```
## License

**MIT License**  
Copyright (c) 2025 Product Recommender AI

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included  
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.

## Usage Restrictions

While this project is open-source under the MIT License, please note the following:

- **Academic and Research Use**: If using this project for academic or research purposes, please cite this repository appropriately.  
- **Commercial Use**: Commercial use is permitted under the MIT License, but we appreciate acknowledgment of the original work.  
- **Data Privacy**: If using this project with real user data, ensure compliance with relevant data protection regulations (e.g., GDPR, CCPA).  
- **Ethical Considerations**: Recommendation systems can influence user behavior. Please use this technology responsibly and consider potential biases in your implementations.  
