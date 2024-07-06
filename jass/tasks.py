

class Task:
    def __init__(self, name, model, instructions, agent=None):
        self.name = name
        self.model = model  # Assume this is an instance of OpenAIModel
        self.instructions = instructions
        self.agent = agent

    def execute(self):
        prompt = self.instructions  # Use the instructions as the prompt
        if self.agent:
            prompt = f"{self.agent.prompt_persona}\n{self.instructions}"
        generated_text = self.model.generate_text(prompt)
        return {'task_output': generated_text}
