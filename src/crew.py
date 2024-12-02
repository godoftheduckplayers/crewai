from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, after_kickoff, before_kickoff
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool


@CrewBase
class CrewaiFinAgent():
    """CrewaiFinAgent crew"""

    @before_kickoff
    def before_kickoff_function(self, inputs):
        print(f"Before kickoff function with inputs: {inputs}")
        return inputs

    @after_kickoff
    def after_kickoff_function(self, result):
        print(f"After kickoff function with result: {result}")
        return result

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='output/report.md'  # This is the file that will be contain the final report.
        )

    # Define the supervisor agent separately to handle coordination
    supervisor_agent = Agent(
        role="Supervisor",
        goal="Leverage the skills of your coworkers to answer the user's query: {query}.",
        backstory="""You are a manager who oversees the workflow of the financial assistant, 
        skilled in overseeing complex workers with different skills and ensuring that you can answer the user's query with the help of the coworkers. 
        You always try to gather data using the financial data agent and / or web scraping agent first.
        After gathering the data, you must delegate to the output summarizing agent to create a comprehensive report instead of answering the user's query directly.""",
        verbose=True,
        allow_delegation=True,
    )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiFinAgent crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,  # creates memory files in "CREWAI_STORAGE_DIR" folder
            verbose=True,  # necessary for memory
            # embedder={
            #     "provider": "azure",
            #     "config": {
            #         "api_key": "eee842323c1e4556b1a7f0ddef120c5a",
            #         "model_name": "azure/text-embedding-3-small",
            #         "deployment": "azure/text-embedding-3-small",
            #         "azure_endpoint": "https://supervisor-ai.openai.azure.com/openai/deployments/text-embedding-3-small",
            #         "api_base": "https://supervisor-ai.openai.azure.com/openai/deployments/text-embedding-3-small",
            #         "api_version": "2023-05-15"
            #     }
            # }
        )
