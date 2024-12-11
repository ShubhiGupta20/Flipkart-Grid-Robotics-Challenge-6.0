# import pandas as pd
# import sqlite3
# from datetime import datetime
# import os

# def save_results(task, result):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     data = [[timestamp, brand, expiry, freshness, count]]
#     # Save to SQLite
#     conn = sqlite3.connect("results.db")
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS results (
#             Timestamp TEXT,
#             Brand TEXT,
#             Expiry TEXT,
#             Freshness TEXT,
#             Count TEXT,
#         )
#     """)
#     conn.commit()
#     pd.DataFrame(data, columns=["Timestamp", "Brand", "Expiry", "Freshness", "Count"]).to_sql("results", conn, if_exists="append", index=False)
#     conn.close()

#     # Save to Excel
#     excel_file = "results.xlsx"
#     if not os.path.exists(excel_file):
#         pd.DataFrame(columns=["Timestamp", "Brand", "Expiry", "Freshness", "Count"]).to_excel(excel_file, index=False)
#     existing = pd.read_excel(excel_file)
#     updated = pd.concat([existing, pd.DataFrame(data, columns=["Timestamp", "Brand", "Expiry", "Freshness", "Count"])])
#     updated.to_excel(excel_file, index=False)





import pandas as pd
import sqlite3
from datetime import datetime
import os

def save_results(task, brand, expiry, freshness, count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = [[timestamp, brand, expiry, freshness, count]]

    # Save to SQLite
    conn = sqlite3.connect("results.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            Timestamp TEXT,
            Brand TEXT,
            Expiry TEXT,
            Freshness TEXT,
            Count TEXT
        )
    """)
    conn.commit()
    pd.DataFrame(data, columns=["Timestamp", "Brand", "Expiry", "Freshness", "Count"]).to_sql("results", conn, if_exists="append", index=False)
    conn.close()

    # Save to Excel
    excel_file = "results.xlsx"
    if not os.path.exists(excel_file):
        # Create the file with proper headers if it doesn't exist
        pd.DataFrame(columns=["Timestamp","Brand", "Expiry", "Freshness", "Count"]).to_excel(excel_file, index=False)

    # Load existing data from the Excel file
    existing = pd.read_excel(excel_file)

    # Append new data
    updated = pd.concat([existing, pd.DataFrame(data, columns=["Timestamp","Brand", "Expiry", "Freshness", "Count"])]).reset_index(drop=True)

    # Save the updated data back to the Excel file
    updated.to_excel(excel_file, index=False)

# Example usage
# save_results("Inventory Check", "BrandA", "2024-12-31", "Fresh", "10")
