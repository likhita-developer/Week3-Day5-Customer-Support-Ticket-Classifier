import pandas as pd
import random
import os

random.seed(42)

categories = {
    "Login Issue": [
        "cannot log into account",
        "password is not working",
        "unable to sign in",
        "login page is rejecting password",
        "forgot password and need help"
    ],
    "Payment Issue": [
        "payment failed",
        "payment was declined",
        "card payment did not go through",
        "charged but payment shows failed",
        "problem with payment"
    ],
    "Refund Request": [
        "want to request a refund",
        "need money back",
        "refund has not arrived",
        "refund is still pending",
        "need refund for my order"
    ],
    "Account Issue": [
        "need help updating account",
        "account details are incorrect",
        "cannot change account information",
        "need to update profile",
        "problem with account"
    ],
    "Technical Issue": [
        "application is not working",
        "website keeps crashing",
        "getting an error in application",
        "page is not loading",
        "system is showing an error"
    ],
    "Order Issue": [
        "order has not arrived",
        "want to know order status",
        "order is delayed",
        "received the wrong order",
        "problem with my order"
    ]
}

prefixes = [
    "Please help me",
    "I need assistance because",
    "I am facing an issue where",
    "Could you please help",
    "I would like support because",
    "I need your help with",
    "Can someone assist me with",
    "I am unable to resolve this issue because",
    "Please check this problem",
    "I want to report that"
]

suffixes = [
    "Please check and resolve it.",
    "This problem started today.",
    "Please help me as soon as possible.",
    "I need this issue resolved.",
    "Can you look into this?",
    "Please provide a solution.",
    "I have been facing this problem recently.",
    "Kindly check my account.",
    "I would appreciate your help.",
    "Please investigate this issue."
]

rows = []

for category, messages in categories.items():
    for i in range(100):
        message = random.choice(messages)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        text = f"{prefix} {message}. {suffix}"

        if i % 10 == 0:
            text = "  " + text.upper() + "  "

        if category in ["Payment Issue", "Refund Request"] and i % 4 == 0:
            priority = "High"
        elif i % 3 == 0:
            priority = "Medium"
        else:
            priority = "Low"

        rows.append({
            "text": text,
            "category": category,
            "priority": priority
        })

df = pd.DataFrame(rows)

# Add 20 duplicate rows intentionally
df = pd.concat([df, df.iloc[:20]], ignore_index=True)

print("Total rows before cleaning:", len(df))

# Clean text
df["text"] = df["text"].str.strip().str.lower()

# Remove duplicate tickets
df = df.drop_duplicates(
    subset=["text", "category", "priority"]
).reset_index(drop=True)

print("Rows after cleaning:", len(df))

# Save cleaned dataset
os.makedirs("data", exist_ok=True)

df.to_csv("data/support_tickets_clean.csv", index=False)

print("Clean dataset saved successfully")
from sklearn.model_selection import train_test_split

# Create stratification key using both labels
df["stratify_key"] = df["category"] + "_" + df["priority"]

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["stratify_key"]
)

# Second split: 15% validation, 15% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["stratify_key"]
)

# Remove helper column
train_df = train_df.drop(columns=["stratify_key"])
validation_df = validation_df.drop(columns=["stratify_key"])
test_df = test_df.drop(columns=["stratify_key"])

# Save datasets
train_df.to_csv("data/train.csv", index=False)
validation_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
