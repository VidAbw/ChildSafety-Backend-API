import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "http://127.0.0.1:8000/api/rag/query"
data = {
    "description": "දරුවෙකුට අන්තර්ජාලය හරහා තර්ජනය කර, පෞද්ගලික ඡායාරූප වෙනත් අය සමඟ බෙදා හරින බව පවසා ඇත",
    "language": "si"
}

req = urllib.request.Request(
    url, 
    data=json.dumps(data).encode("utf-8"), 
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print(json.dumps(res, indent=2, ensure_ascii=False))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")
