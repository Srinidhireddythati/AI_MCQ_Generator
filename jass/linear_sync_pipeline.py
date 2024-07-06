class LinearSyncPipeline:
    def __init__(self, name, completion_message, tasks):
        self.name = name
        self.completion_message = completion_message
        self.tasks = tasks

    def run(self):
        outputs = []
        for task in self.tasks:
            task_output = task.execute()
            outputs.append(task_output)
        return outputs
