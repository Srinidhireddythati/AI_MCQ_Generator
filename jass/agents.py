class Agent:
    def __init__(self, role, prompt_persona):
        self.role = role
        self.prompt_persona = prompt_persona
    def get_role(self):
        return self.role

    def get_prompt_persona(self):
        return self.prompt_persona