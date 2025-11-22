import pandas as pd

data = {
    "customer_name": ["John Smith", "Jane Doe", "Robert Brown", "Emily Davis", "Michael Wilson", "Sarah Miller", "John Smith", "David Lee", "Laura White", "John Smith", "Jessica Chen", "Mark Taylor"],
    "email": ["john.smith@example.com", "jane.doe@webmail.com", "rob.brown@email.com", "emily@company.com", "mike.w@example.com", "s.miller@webmail.com", "jsmith2@personal.com", "d.lee@company.com", "laura.white@email.com", "smith.john@fastmail.com", "jess.chen@example.com", "mark.t@webmail.com"],
    "age": [65, 22, 45, 30, 50, 28, 33, 42, 55, 29, 38, 61],
    "plan_type": ["Gold", "Silver", "Silver", "Bronze", "Gold", "Bronze", "Silver", "Gold", "Platinum", "Bronze", "Platinum", None],
    "monthly_spend": [150.00, 70.50, 80.00, None, 140.00, 45.00, 75.00, 160.00, 999.00, 50.00, 210.00, 135.00],
    "last_login_days_ago": [5, 1, 20, 2, 8, 3, 30, 1, 12, 45, 4, 9],
    "has_churned": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)
df.to_csv("valid_test_data.csv", index=False)
print("SUCCESS! Created 'valid_test_data.csv' with data.")