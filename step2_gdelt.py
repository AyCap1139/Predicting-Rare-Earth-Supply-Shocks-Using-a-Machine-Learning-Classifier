from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="ree-classifier")

query = """
SELECT
    CAST(SQLDATE AS STRING) AS date,
    COUNT(*) AS event_count,
    AVG(AvgTone) AS avg_tone,
    AVG(GoldsteinScale) AS avg_goldstein
FROM
    `gdelt-bq.gdeltv2.events`
WHERE
    SQLDATE >= 20200101
    AND SQLDATE <= 20251231
    AND Actor1CountryCode = 'CHN'
    AND EventCode IN ('061','163','193','172','173')
GROUP BY
    date
ORDER BY
    date ASC
"""

print("Running GDELT query — this may take 30-60 seconds...")
df = client.query(query).to_dataframe()
print(f"Done. Got {len(df)} rows.")
print(df.head(10))

df.to_csv("gdelt_ree_daily.csv", index=False)
print("Saved to gdelt_ree_daily.csv")