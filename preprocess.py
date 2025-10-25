# Step 2: Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical columns
label_enc = LabelEncoder()
df["room_type"] = label_enc.fit_transform(df["room_type"])
df["status"] = label_enc.fit_transform(df["status"])  # Target

# Split features and target
X = df.drop("status", axis=1)
y = df["status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data preprocessing complete!")
