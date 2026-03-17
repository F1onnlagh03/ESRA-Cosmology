import lasair

TOKEN = ""  # exactly as given by Lasair, no extra spaces
client = lasair.lasair_client(TOKEN)

# Test a minimal query
try:
    test = client.query(selected="objectId", tables="objects", conditions="1=1", limit=1)
    print("Token works! Result:", test)
except lasair.LasairError as e:
    print("Error:", e)