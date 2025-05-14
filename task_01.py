import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

def true_data(df):
    return df[["consumption_kwh"]].values

def read_data():
    df = pd.read_csv('daily_energy_consumption.csv')
    df["date"] = pd.to_datetime(df["date"])
    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)

    return df

def load_training_data(df, seed=42):
    np.random.seed(seed)

    X_train = scaler_X.fit_transform(df[["date_ordinal"]].values)
    y_train = scaler_y.fit_transform(true_data(df))

    return X_train, y_train

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    return model

def evaluate_model(model, X_test):
    return model.predict(X_test)

def train_scalers():
    scaler_X.fit(df[["date_ordinal"]])  # Навчити повторно на тих самих даних
    scaler_y.fit(df[["consumption_kwh"]])

def inverse_transform_X(X):
    X_ordinal = scaler_X.inverse_transform(X).flatten()

    return X_ordinal

def convert_to_date(X_ordinal):
    X_dates = [pd.Timestamp.fromordinal(int(i)) for i in X_ordinal]

    return X_dates

def inverse_transform_y(y):
    y_inv = scaler_y.inverse_transform(y)

    return y_inv

def get_true_values(X_test_dates):
    df_indexed = df.set_index("date")
    y_true = []

    for date in X_test_dates:
        if date in df_indexed.index:
            y_true.append(df_indexed.loc[date, "consumption_kwh"])
        else:
            y_true.append(np.nan)  # Якщо немає такого дня — NaN

    return y_true

def inverse_transform(X_train, X_test, y_train, y_pred):
    X_train_ordinal = inverse_transform_X(X_train)
    X_test_ordinal = inverse_transform_X(X_test)

    y_train_inv = inverse_transform_y(y_train)
    y_pred_inv = inverse_transform_y(y_pred)

    return X_train_ordinal, X_test_ordinal, y_train_inv, y_pred_inv

def convert_all_to_date(X_train, X_test):
    X_train_ordinal = inverse_transform_X(X_train)
    X_test_ordinal = inverse_transform_X(X_test)
    
    X_train_dates = convert_to_date(X_train_ordinal)
    X_test_dates = convert_to_date(X_test_ordinal)

    return X_train_dates, X_test_dates

def inverse_transform_y_all(y_train, y_pred):
    y_train_inv = inverse_transform_y(y_train)
    y_pred_inv = inverse_transform_y(y_pred)

    return y_train_inv, y_pred_inv

def visualize_results(X_train, y_train, X_test, y_pred, y_true):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_dates, y_train_inv, label='Training Data', alpha=0.3)
    plt.plot(X_test_dates, y_true, color='green', label='True Values', linestyle="dashed")
    plt.plot(X_test_dates, y_pred_inv, color='red', label='NN Prediction')
    plt.title('Energy Consumption Prediction')
    plt.xlabel('Date')
    plt.ylabel('Consumption (kWh)')
    plt.legend()
    plt.grid()
    plt.show()

def predict(model, X_test, scaler_X, scaler_y):
    X_input = scaler_X.transform(np.array([[X_test]]))
    y_pred = model.predict(X_input)
    return scaler_y.inverse_transform(y_pred)[0, 0]

if __name__ == "__main__":
    df = read_data()
    X_train, y_train = load_training_data(df)
    
    X_train = scaler_X.fit_transform(df[["date_ordinal"]].values)
    y_train = scaler_y.fit_transform(true_data(df))

    X_test = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100).reshape(-1, 1)

    model = build_model()

    train_model(model, X_train, y_train, epochs=100)

    y_pred = evaluate_model(model, X_test)

    train_scalers()

    X_train_dates, X_test_dates = convert_all_to_date(X_train, X_test)
    y_train_inv, y_pred_inv = inverse_transform_y_all(y_train, y_pred)
    y_true = get_true_values(X_test_dates)

    visualize_results(X_train_dates, y_train_inv, X_test_dates, y_pred_inv, y_true)

    x_new_ordinal = 738791
    y_new = predict(model, x_new_ordinal, scaler_X, scaler_y)
    x_new_date = convert_to_date([x_new_ordinal])[0]

    print(f"Predicted consumption for date = {x_new_date}: {y_new}")