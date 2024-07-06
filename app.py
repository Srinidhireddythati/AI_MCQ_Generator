import streamlit as st
from jass.openai_model import OpenAIModel
from jass.agents import Agent
from jass.tasks import Task
from jass.linear_sync_pipeline import LinearSyncPipeline


# Set Streamlit page configuration
st.set_page_config(
    page_title="MCQ Generator",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS styles for Streamlit components
st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI components
st.title("MCQ Generator")
st.markdown("### Welcome to the MCQ Generator!")
st.markdown("Upload Your Topic and get Perfect Answers.")

# Input fields for API key, topic, and number of questions
api_key = st.text_input("Enter OpenAI API Key")
topic = st.text_input("Enter Topic")
question_limit = st.number_input("Enter number of MCQ questions", min_value=1, step=1)

# Process input if all fields are provided
if topic and api_key and question_limit:
    try:
        # Initialize OpenAIModel with provided API key and parameters
        open_ai_text_completion_model = OpenAIModel(
            api_key=api_key,
            parameters={
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 1500,
            },
        )

        ielts_agent = Agent(
            role="Ielts expert",
            prompt_persona=f"Your task is to DEVELOP {question_limit} MULTIPLE-CHOICE QUESTIONS (MCQ) about {topic} and also give their answers"
        )

        agent = Agent(role="assistant", prompt_persona="friendly")

        ielts_task = Task(
            name="Generate IELTS MCQs",
            model=open_ai_text_completion_model,
            agent=ielts_agent,
            instructions=f"Give {question_limit} MCQ Questions with answers",
        )

        # Define a linear synchronous pipeline for the task
        pipeline = LinearSyncPipeline(
            name="Ielts details",
            completion_message="pipeline completed",
            tasks=[ielts_task],
        )

        # Execute the pipeline and capture the output
        output = pipeline.run()

        # Display generated MCQs if pipeline output is available
        if output:
            st.markdown("## Generated MCQs")
            st.markdown(output[0]['task_output'])

    except Exception as e:
        # Display error message if pipeline execution fails
        st.error(f"Error running the pipeline: {e}")
