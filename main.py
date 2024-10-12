import gradio as gr
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import json

print("Imports successful")

# State definition
class State(TypedDict):
    user_data: dict
    fitness_plan: str
    feedback: str
    progress: List[str]
    messages: Annotated[list, add_messages]

print("State class defined")

# Utility function to get Ollama LLM
def get_ollama_llm(model_name="tinyllama"):
    print(f"Creating ChatOllama with model: {model_name}")
    return ChatOllama(model=model_name)

# User Input Agent
def user_input_agent(state: State, llm):
    print("Entering user_input_agent")
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach assistant. Process the following user information:

        {user_input}

        Create a structured user profile based on this information. Include all relevant details for creating a personalized fitness plan.
        Return the profile as a valid JSON string."""
    )
    chain = prompt | llm | StrOutputParser()
    print(f"User input: {state['user_data']}")
    user_profile = chain.invoke({"user_input": json.dumps(state["user_data"])})
    print(f"Generated user profile: {user_profile}")
    try:
        state["user_data"] = json.loads(user_profile)
    except json.JSONDecodeError:
        print("Failed to parse user profile as JSON")
        # If JSON parsing fails, keep the original user_data
        pass
    state["messages"].append(AIMessage(content=f"Processed user profile: {json.dumps(state['user_data'], indent=2)}"))
    print("Exiting user_input_agent")
    return state

# Routine Generation Agent
def routine_generation_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach. Create a personalized fitness routine based on the following user data:

        {user_data}

        Create a detailed weekly fitness plan that includes:
        1. Types of exercises
        2. Duration and frequency of workouts
        3. Intensity levels
        4. Rest days
        5. Any dietary recommendations

        Present the plan in a clear, structured format."""
    )
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"user_data": json.dumps(state["user_data"])})
    state["fitness_plan"] = plan
    state["messages"].append(AIMessage(content=f"Generated fitness plan: {plan}"))
    return state

# Feedback Collection Agent
def feedback_collection_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach assistant. Analyze the following user feedback on their recent workout session:

        Current fitness plan: {current_plan}
        User feedback: {user_feedback}

        Summarize the user's feedback and suggest any immediate adjustments."""
    )
    chain = prompt | llm | StrOutputParser()
    feedback_summary = chain.invoke({"current_plan": state["fitness_plan"], "user_feedback": state["feedback"]})
    state["messages"].append(AIMessage(content=f"Feedback analysis: {feedback_summary}"))
    return state

# Routine Adjustment Agent
def routine_adjustment_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach. Adjust the current fitness plan based on the user's feedback:

        Current Plan:
        {current_plan}

        User Feedback:
        {feedback}

        Provide an updated weekly fitness plan that addresses the user's feedback while maintaining the overall structure and goals."""
    )
    chain = prompt | llm | StrOutputParser()
    updated_plan = chain.invoke({"current_plan": state["fitness_plan"], "feedback": state["feedback"]})
    state["fitness_plan"] = updated_plan
    state["messages"].append(AIMessage(content=f"Updated fitness plan: {updated_plan}"))
    return state

# Progress Monitoring Agent
def progress_monitoring_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness progress tracker. Review the user's progress and provide encouragement or suggestions:

        User Data: {user_data}
        Current Plan: {current_plan}
        Progress History: {progress_history}

        Provide a summary of the user's progress, offer encouragement, and suggest any new challenges or adjustments."""
    )
    chain = prompt | llm | StrOutputParser()
    progress_update = chain.invoke(
        {"user_data": str(state["user_data"]), "current_plan": state["fitness_plan"], "progress_history": str(state["progress"])}
    )
    state["progress"].append(progress_update)
    state["messages"].append(AIMessage(content=f"Progress update: {progress_update}"))
    return state

# Motivational Agent
def motivational_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI motivational coach for fitness. Provide encouragement, tips, or reminders to the user:

        User Data: {user_data}
        Current Plan: {current_plan}
        Recent Progress: {recent_progress}

        Generate a motivational message, helpful tip, or reminder to keep the user engaged and committed to their fitness goals."""
    )
    chain = prompt | llm | StrOutputParser()
    motivation = chain.invoke(
        {"user_data": str(state["user_data"]), "current_plan": state["fitness_plan"], "recent_progress": state["progress"][-1] if state["progress"] else ""}
    )
    state["messages"].append(AIMessage(content=f"Motivation: {motivation}"))
    return state

# AIFitnessCoach class
class AIFitnessCoach:
    def __init__(self):
        print("Initializing AIFitnessCoach")
        self.llm = get_ollama_llm()
        self.graph = self.create_graph()

    def create_graph(self):
        print("Creating graph")
        workflow = StateGraph(State)

        # Define nodes
        workflow.add_node("user_input", lambda state: user_input_agent(state, self.llm))
        workflow.add_node("routine_generation", lambda state: routine_generation_agent(state, self.llm))
        workflow.add_node("feedback_collection", lambda state: feedback_collection_agent(state, self.llm))
        workflow.add_node("routine_adjustment", lambda state: routine_adjustment_agent(state, self.llm))
        workflow.add_node("progress_monitoring", lambda state: progress_monitoring_agent(state, self.llm))
        workflow.add_node("motivation", lambda state: motivational_agent(state, self.llm))

        # Define edges
        workflow.add_edge("user_input", "routine_generation")
        workflow.add_edge("routine_generation", "feedback_collection")
        workflow.add_edge("feedback_collection", "routine_adjustment")
        workflow.add_edge("routine_adjustment", "progress_monitoring")
        workflow.add_edge("progress_monitoring", "motivation")
        workflow.add_edge("motivation", END)

        # Set entry point
        workflow.set_entry_point("user_input")

        print("Graph created")
        return workflow.compile()

    def run(self, user_input):
        print("Running AIFitnessCoach")
        initial_state = State(
            user_data=user_input,
            fitness_plan="",
            feedback="",
            progress=[],
            messages=[HumanMessage(content=json.dumps(user_input))]
        )
        print(f"Initial state: {initial_state}")
        final_state = self.graph.invoke(initial_state)
        print(f"Final state: {final_state}")
        return final_state["messages"]

# Helper function for Gradio interface
def process_user_input(age, weight, gender, primary_goal, target_timeframe, workout_preferences, 
                       workout_duration, workout_days, activity_level, health_conditions, 
                       dietary_preferences):
    print("Processing user input")
    user_data = {
        "age": age,
        "weight": weight,
        "gender": gender,
        "primary_goal": primary_goal,
        "target_timeframe": target_timeframe,
        "workout_preferences": workout_preferences,
        "workout_duration": workout_duration,
        "workout_days": workout_days,
        "activity_level": activity_level,
        "health_conditions": health_conditions,
        "dietary_preferences": dietary_preferences
    }
    print(f"User data: {user_data}")
    coach = AIFitnessCoach()
    messages = coach.run(user_data)
    print(f"Generated messages: {messages}")
    return "\n\n".join([f"{message.type.capitalize()}: {message.content}" for message in messages])

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# AI Fitness Coach")
    
    with gr.Tab("Create Fitness Plan"):
        with gr.Row():
            age = gr.Number(label="Age")
            weight = gr.Number(label="Weight (kg)")
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
        
        primary_goal = gr.Dropdown(["Weight loss", "Muscle gain", "Endurance improvement", "General fitness"], label="Primary Goal")
        target_timeframe = gr.Dropdown(["3 months", "6 months", "1 year"], label="Target Timeframe")
        
        workout_preferences = gr.CheckboxGroup(
            ["Cardio", "Strength training", "Yoga", "Pilates", "Flexibility exercises", "HIIT"],
            label="Workout Type Preferences"
        )
        workout_duration = gr.Slider(15, 120, step=15, label="Preferred Workout Duration (minutes)")
        workout_days = gr.CheckboxGroup(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            label="Preferred Workout Days"
        )
        
        activity_level = gr.Radio(
            ["Sedentary", "Lightly active", "Moderately active", "Highly active"],
            label="Current Activity Level"
        )
        health_conditions = gr.Textbox(label="Health Conditions or Injuries")
        dietary_preferences = gr.Textbox(label="Dietary Preferences (Optional)")
        
        create_button = gr.Button("Create Fitness Plan")
        plan_output = gr.Textbox(label="Your Personalized Fitness Plan")

        create_button.click(
            process_user_input,
            inputs=[age, weight, gender, primary_goal, target_timeframe, workout_preferences, 
                    workout_duration, workout_days, activity_level, health_conditions, 
                    dietary_preferences],
            outputs=plan_output
        )

    with gr.Tab("Update Fitness Plan"):
        feedback = gr.Textbox(label="Your Feedback")
        update_button = gr.Button("Update Fitness Plan")
        updated_plan_output = gr.Textbox(label="Updated Fitness Plan")

        def update_plan(feedback):
            print(f"Updating plan with feedback: {feedback}")
            coach = AIFitnessCoach()
            messages = coach.run({"feedback": feedback})
            print(f"Updated messages: {messages}")
            return "\n\n".join([f"{message.type.capitalize()}: {message.content}" for message in messages])

        update_button.click(update_plan, inputs=[feedback], outputs=updated_plan_output)

print("Gradio UI defined")

if __name__ == "__main__":
    print("Launching Gradio demo")
    demo.launch()