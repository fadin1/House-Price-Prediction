from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = [
    [800, 2, 10],
    [1000, 2, 8],
    [1200, 3, 5],
    [1500, 3, 5],
    [1800, 4, 3],
    [2000, 4, 2],
    [2200, 4, 1],
    [900, 2, 12],
    [1400, 3, 7],
    [1700, 3, 4],
    [1100, 2, 6],
    [2400, 5, 2]
]

y = [25, 30, 38, 45, 55, 65, 70, 27, 42, 52, 34, 78]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f} lakhs")

size = float(input("Enter house size in sq ft: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = float(input("Enter house age in years: "))

new_house = [[size, bedrooms, age]]
predicted_price = model.predict(new_house)

print("\n--- House Price Prediction ---")
print(f"House size: {size} sq ft")
print(f"Bedrooms: {bedrooms}")
print(f"House age: {age} years")
print(f"Predicted house price: ₹{predicted_price[0]:.2f} lakhs")
