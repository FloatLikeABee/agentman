import requests
import json


api_key = "0aa9ad7dd2114ef3bfb16bea056a977a.9nVmCTUa8l6SBCK6"
api_base_url = "https://api.z.ai/api/paas/v4"
model = 'glm-4.6'
temperature = 1.0
max_tokens = 1024

prompt = """
tell me about the end of the world 2012 rumor
"""

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept-Language": "en-US,en",
    "Content-Type": "application/json"
}
def main_one():
    try:
        url = f"{api_base_url}/chat/completions"

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(content)
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.RequestException as e:
        print(f"GLM streaming request failed: {e}")



def call_zai_api(messages, model="glm-4.6"):
    url = "https://api.z.ai/api/paas/v4/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en"
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed: {response.status_code}, {response.text}")

# Usage example
messages = [
    {"role": "user", "content": f"""Hello, {prompt}"""}
]

result = call_zai_api(messages)
print(result['choices'][0]['message']['content'])
