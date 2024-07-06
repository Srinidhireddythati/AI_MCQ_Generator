import openai

class OpenAIModel:
    def __init__(self, api_key, parameters):
        self.api_key = api_key
        self.parameters = parameters
        openai.api_key = api_key
    
    def generate_text(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.parameters.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.parameters.get("max_tokens", 1500),
                temperature=self.parameters.get("temperature", 0.2)
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"
