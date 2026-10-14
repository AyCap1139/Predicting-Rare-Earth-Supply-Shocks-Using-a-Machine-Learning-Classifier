from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="ree-classifier")

query = """
SELECT
    CAST(SQLDATE AS STRING) AS date,
    COUNT(*) AS china_media_count,
    AVG(AvgTone) AS china_media_tone,
    COUNTIF(UPPER(SourceURL) LIKE '%XINHUA%') AS xinhua_count,
    COUNTIF(UPPER(SourceURL) LIKE '%GLOBALTIMES%') AS globaltimes_count,
    COUNTIF(UPPER(SourceURL) LIKE '%MOFCOM%') AS mofcom_count,
    COUNTIF(UPPER(SourceURL) LIKE '%PEOPLE%') AS peopledaily_count
FROM
    `gdelt-bq.gdeltv2.events`
WHERE
    SQLDATE >= 20200101
    AND SQLDATE <= 20251231
    AND Actor1CountryCode = 'CHN'
    AND EventCode IN ('061','163','193','172','173')
    AND (
        UPPER(SourceURL) LIKE '%XINHUA%'
        OR UPPER(SourceURL) LIKE '%GLOBALTIMES%'
        OR UPPER(SourceURL) LIKE '%MOFCOM%'
        OR UPPER(SourceURL) LIKE '%PEOPLE%'
        OR UPPER(SourceURL) LIKE '%CHINA%'
    )
GROUP BY
    date
ORDER BY
    date ASC
"""

print("Running Chinese media GDELT query...")
df = client.query(query).to_dataframe()
print(f"Done. Got {len(df)} rows.")
print(df.head(10))

df.to_csv("chinese_media_daily.csv", index=False)
print("Saved to chinese_media_daily.csv")