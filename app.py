import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# # Uploading Datay


def upload_data():
    choice = input("Do you want to upload a CSV file (y/n)? ").lower()

    if choice == 'y':
        file_path = input("Enter the path of the CSV file: ")
        data = pd.read_csv(file_path)
    else:
        # Manual data entry
        data = {'X': [], 'y': []}
        n_samples = int(input("Enter the number of data points: "))

        # Loop through each data point and input values for X and y
        for i in range(n_samples):
            x_value = float(input(f"Enter the value for X({i+1}): "))
            y_value = float(input(f"Enter the corresponding value for y({i+1}): "))

            # Append the entered values to the data dictionary
            data['X'].append(x_value)
            data['y'].append(y_value)

        # Convert the data dictionary to a DataFrame
        data = pd.DataFrame(data)

    # Return the resulting DataFrame containing the uploaded or manually entered data
    return data


# # Linear Regression

def perform_linear_regression(data):
    # Check if 'X' and 'y' columns are present in the dataset
    if 'X' not in data.columns or 'y' not in data.columns:
        print("Error: 'X' or 'y' column not found in the dataset.")
        return

    # Extract independent variable X and dependent variable y
    X = data[['X']]
    y = data['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict the target variable on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the regression line
    plt.scatter(X_test, y_test, color='black', label='Actual')
    plt.scatter(X_test, y_pred, color='blue', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Linear Regression (MSE: {mse:.2f})')
    plt.legend()
    plt.show()
    
    
 #performing linear regression   
if __name__ == "__main__":
    print("Welcome to the Machine Learning Application!")

    # Step 1: Upload data
    data = upload_data()

    # Step 2: Perform linear regression analysis
    perform_linear_regression(data)
    