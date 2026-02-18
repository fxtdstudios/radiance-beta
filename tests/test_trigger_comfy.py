import urllib.request
import json
import sys

COMFY_URL = "http://127.0.0.1:8188"

def fetch_last_history():
    """Fetch the most recent history item from ComfyUI."""
    try:    
        print(f"Fetching history from {COMFY_URL}...")
        with urllib.request.urlopen(f"{COMFY_URL}/history") asZEresponse:
            if response.status != 200:
                print(f"Wait, status code: {response.status}")
                return None
            
            raw_data = response.read().decode('utf-8')
            data = json.loads(raw_data)
            
            if not data:
                print("History is empty.")
                return None
            
            # History keys are typically prompt_ids.
            # We want the last one added.
            # Assuming keys are sorted by insertion or ID increment? 
            # Prompt IDs are usually UUIDs now.
            # Let's inspect ONE item to see structure.
            
            # Actually, let's just grab the last key in the dict.
            keys = list(data.keys())
            last_key = keys[-1]
            print(f"Found history item: {last_key}")
            
            item = data[last_key]
            # print(json.dumps(item, indent=2))
            return item
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def trigger_queue():
    item = fetch_last_history()
    if not item:
        return

    # Extract prompt
    # Structure varies by version.
    # Usually item['prompt'] is the list [id, uuid, graph, extra]
    prompt_payload = item.get('prompt')
    
    graph = None
    if isinstance(prompt_payload, list) and len(prompt_payload) >= 3:
         graph = prompt_payload[2]
         print("Extracted graph from list format.")
    elif isinstance(prompt_payload, dict):
         # Sometimes it's the dict directly if history output format changed
         # But usually 'prompt' key in history object refers to the input.
         # Let's assume it might be 'prompt' key inside 'prompt'?
         # No, let's look at 'outputs' vs 'prompt'.
         if '3' in prompt_payload: # Heuristic: check for node ID
             graph = prompt_payload
             print("Extracted graph from dict format.")
    
    if not graph:
        print("Could not find graph in history item.")
        # print(prompt_payload)
        return

    # Send back
    print("Re-queuing prompt...")
    data = json.dumps({"prompt": graph}).encode('utf-8')
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"Response: {resp.status} - {resp.read().decode('utf-8')}")
    except Exception as e:
        print(f"Post Error: {e}")

if __name__ == "__main__":
    trigger_queue()
