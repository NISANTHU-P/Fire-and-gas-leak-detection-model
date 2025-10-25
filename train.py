# Step 3: Random Forest Model Training
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
clf.fit(X_train, y_train)

print("Random Forest training complete!")
