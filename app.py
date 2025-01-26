from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


app = Flask(__name__)

MODEL_ = None
SCALER = None

user_input = dict()

@app.route('/')
def index():
    return render_template('index-2.html')


def model_loading():

    global MODEL_, SCALER

    df = pd.read_csv("/Users/pranavmody/Downloads/detailed_us_records_20k.csv")



    cols_toRemove = ['First Name','Last Name', 'Address', 'SSN Number' ,
                    'Country', 'Phone Number','Company']

    df = df.drop(columns=cols_toRemove)

    print(df.columns)

    categorical_cols = ['Home Owner Status','Employment Title','Bank Name', 'Occupation', 'Loan Purpose']

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols)



    df[['Low', 'High']] = df['Assets and Properties Amount Range'].str.split(' - ', expand=True)

    # Convert the new columns to numeric
    df[['Low', 'High']] = df[['Low', 'High']].replace('[\$,]', '', regex=True).astype(float)

    # Calculate the average
    df['Average'] = (df['Low'] + df['High']) / 2

    df = df.drop(columns=['Assets and Properties Amount Range'])


    class_map = {'Accepted': 0, 'Rejected': 1}
    df['Loan Status'] = df['Loan Status'].map(class_map)







    X = df.drop(columns=['Loan Status'])
    y = df['Loan Status']

    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, stratify=y, random_state=42)


    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(10), max_iter=300, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    SCALER = scaler

    MODEL_ = model


def get_predictions():
    '''scaler = StandardScaler()
    X_test_scaled = scaler.transform(user_input)
    y_pred = MODEL_.predict(user_input)'''
    if user_input['annual_income'] < 50000:
        return 0
    else:
        return 1
    

@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/register.html',methods=['GET', 'POST'])
def register():
    print("registered called")
    return render_template('register.html')

def assign_interest_rate(row):
    # Base interest rate assumptions
    base_rate = 5.0  # Starting with a base interest rate of 5%
    
    # Adjust based on FICO score
    if row["FICO Score"] >= 801:
        rate_adjustment = -1.0
    elif 701 <= row["FICO Score"] <= 800:
        rate_adjustment = -0.5
    else:
        rate_adjustment = 0.0  # No change for scores below 700 in this simplified model
    
    # Further adjust based on loan purpose
    if row["Loan Purpose"] == "Home":
        rate_adjustment -= 0.5
    elif row["Loan Purpose"] == "Personal":
        rate_adjustment += 1.0
    # Education loans remain neutral in this example, no additional adjustment
    
    # Final interest rate calculation
    final_rate = base_rate + rate_adjustment
    
    # Ensuring the final rate doesn't go below a minimum threshold
    final_rate = max(3.0, final_rate)  # Assuming a minimum interest rate of 3%
    
    return final_rate


def my_dash():
    df = pd.read_csv("/Users/pranavmody/Downloads/detailed_us_records_20k.csv")
    df = df[df['Loan Status'] == 'Accepted']
    dict_ = user_input
    selected_columns = df[list(dict_.keys())]
    categorical_cols = ['Home Owner Status','Employment Title','Bank Name','Loan Purpose']
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        selected_columns[col] = label_encoders[col].fit_transform(selected_columns[col])
    selected_columns[['Low', 'High']] = selected_columns['Assets and Properties Amount Range'].str.split(' - ', expand=True)

    # Convert the new columns to numeric
    selected_columns[['Low', 'High']] = selected_columns[['Low', 'High']].replace('[\$,]', '', regex=True).astype(float)

    # Calculate the average
    selected_columns['Assets and Properties Amount Range'] = (selected_columns['Low'] + selected_columns['High']) / 2

    selected_columns.drop(columns='High',inplace=True)
    selected_columns.drop(columns='Low',inplace=True)

    dict_['Home Owner Status'] = label_encoders['Home Owner Status'].transform([dict_['Home Owner Status']])[0]
    dict_['Employment Title'] = label_encoders['Employment Title'].transform([dict_['Employment Title']])[0]
    dict_['Loan Purpose'] = label_encoders['Loan Purpose'].transform([dict_['Loan Purpose']])[0]
    dict_['Bank Name'] = label_encoders['Bank Name'].transform([dict_['Bank Name']])[0]

    k = 3  # Number of neighbors to find
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(selected_columns)

    # Find the 3 nearest neighbors for the input data
    distances, indices = knn.kneighbors(pd.DataFrame([dict_]))

    # Get the indices of the 3 nearest neighbors
    nearest_neighbors_indices = indices[0]

    # Get the 3 nearest neighbors from selected_columns
    nearest_neighbors = selected_columns.iloc[nearest_neighbors_indices]

    

    original_df = pd.read_csv("/content/detailed_us_records_20k.csv")

    out1_inc = original_df.iloc[nearest_neighbors.index[0]]['annual_income']
    out2_inc = original_df.iloc[nearest_neighbors.index[1]]['annual_income']
    out3_inc = original_df.iloc[nearest_neighbors.index[2]]['annual_income']

    out1_rate = assign_interest_rate(original_df.iloc[nearest_neighbors.index[0]]['annual_income'])
    out2_rate = assign_interest_rate(original_df.iloc[nearest_neighbors.index[1]]['annual_income'])
    out3_rate = assign_interest_rate(original_df.iloc[nearest_neighbors.index[2]]['annual_income'])

    selected_columns = selected_columns[dict_['Bank Name'] == selected_columns['Bank Name']]
    k = 1  # Number of neighbors to find
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(selected_columns)

    # Find the 3 nearest neighbors for the input data
    distances, indices = knn.kneighbors(pd.DataFrame([dict_]))

    # Get the indices of the 3 nearest neighbors
    nearest_neighbors_indices = indices[0]

    # Get the 3 nearest neighbors from selected_columns
    nearest_neighbors = selected_columns.iloc[nearest_neighbors_indices]


    for keys in label_encoders.keys():
        nearest_neighbors[keys] = label_encoders[keys].inverse_transform([nearest_neighbors[keys]])[0]
        dict_[keys] = label_encoders[keys].inverse_transform([dict_[keys]])[0]

    out1_inc = original_df.iloc[nearest_neighbors.index[0]]['annual_income']
    out1_rate = assign_interest_rate(original_df.iloc[nearest_neighbors.index[0]]['annual_income'])

    return out1_inc,out2_inc,out3_inc,out1_rate,out2_rate,out3_rate,out1_inc,out1_rate
                                     




@app.route('/hackform_submit.html',methods=['GET', 'POST'])
def hackform_submit():
    #input("hackform submit ??")
    user_input['first_name'] = request.form.get('first_name')
    user_input['last_name'] = request.form.get('last_name')
    user_input['Address'] = request.form.get('Address')
    user_input['Country'] = request.form.get('Country')
    user_input['Zipcode'] = int(request.form.get('Zipcode'))
    user_input['Contact'] = int(request.form.get('Contact'))
    user_input['SSN'] = int(request.form.get('SSN'))
    user_input['family_member'] = int(request.form.get('family_member'))
    user_input['salaried_member'] = int(request.form.get('salaried_member'))
    user_input['bank_name'] = request.form.get('bank_name')
    user_input['annual_income'] = int(request.form.get('annual_income'))
    user_input['company'] = request.form.get('company')
    user_input['occupation'] = request.form.get('occupation')
    user_input['emp_title'] = request.form.get('emp_title')
    user_input['ownership'] = request.form.get('ownership')
    user_input['assets'] = request.form.get('assets')
    user_input['requirements'] = request.form.get('requirements')
    user_input['purpose'] = request.form.get('purpose')
    print("POST called")
    print(user_input)
    out_ = get_predictions()

    if out_ == 1:
        return render_template('/admin/hackdashboard.html')
    else:
        return render_template('/failed.html')


@app.route('/hackform.html',methods=['GET', 'POST'])
def hackform():
    if request.method == 'POST':
        print("Nothing hackform")
    else:
        print("inside register func")
    return render_template('hackform.html')

@app.route('/admin/hackdashboard.html',methods=['GET', 'POST'])
def hackdashboard():
    return render_template('/admin/hackdashboard.html')

@app.route('/failed.html')
def failed():
    return render_template('failed.html')

if __name__ == '__main__':
    app.run(debug=True)

