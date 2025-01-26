# FinAI Navigate - Revolutionizing Loan Paths with Smart Approval Systems

### Project Overview
In today's financial landscape, the loan application process remains daunting, opaque, and inaccessible for many. Applicants face numerous hurdles such as understanding eligibility criteria, navigating various loan types, and coping with the discouragement of loan rejections. This project aims to revolutionize the loan approval process by utilizing AI-driven insights to make financial support more accessible, personalized, and efficient.

### Solution
Our solution leverages cutting-edge machine learning algorithms and dynamic decision engines to analyze financial data and credit history in real-time, predicting loan eligibility with a high degree of accuracy. Key features include:

- **Machine Learning Algorithms**: Sophisticated models to predict loan eligibility based on financial data.
- **Dynamic Decision Engines**: Continuously improving accuracy by learning from each application.
- **Automated Feedback Loops**: Personalized feedback for applicants not approved, offering clear steps for improvement.
- **API Integrations**: Seamless connectivity with banking partners for swift transition from application to approval.

### Tech Stack
- **Python**: Backend development and machine learning model implementation.
- **Pandas**: Data manipulation and preprocessing.
- **Flask**: Web application framework for API development.
- **HTML/CSS**: Frontend development for the user interface.
- **AWS**: Cloud hosting and deployment.
- **Jupyter Notebooks**: Model training and analysis.

### Process

1. **Data Cleaning and Preprocessing**:
    - **Column Removal**: Discard non-essential features using `df.drop(columns=cols_toRemove)`.
    - **Categorical Variable Encoding**: Transform categorical features into binary vectors using `pd.get_dummies()`.
    - **Numeric Conversion**: Encode ranges and target variables for ML model input.
    - **Feature Standardization**: Normalize feature distributions with `StandardScaler()`.

2. **Exploratory Data Analysis (EDA)**:
    - Analysis of the dataset to understand the underlying patterns and correlations.

3. **Model Development**:
    - **Algorithm**: Multi-Layer Perceptron (MLP) Classifier.
    - **Training**: The model was trained using scaled features and backpropagation.
    - **Evaluation**: The model achieved an accuracy score of 81.4%.

### Dataset
- Detailed U.S. financial records from the following source:
  [Detailed US Records 20k CSV](https://loandata2020.s3.us-west-1.amazonaws.com/detailed_us_records_20k.csv)

### Model and Analysis
- Model training and analysis performed in Google Colab.
  [Colab Notebook Link](https://colab.research.google.com/drive/1hXuwo0EWpaVovC4ikXQNpxGt6u2uJh0I?usp=sharing)

### Project Structure

```plaintext
├── static/
│   └── [Static Files]
├── templates/
│   └── [HTML Templates]
├── .DS_Store
├── Group10_Presentation.pptx
├── README.md
├── app.py
└── detailed_us_records_20k.csv
