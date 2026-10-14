from google.cloud import bigquery

client = bigquery.Client(project="ree-classifier")
print("Connected successfully:", client.project)