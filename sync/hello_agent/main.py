import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAgent:
    """Synchronous wrapper for Gemini API interactions."""
    
    def __init__(self):
        self.model, self.config = self._initialize_gemini()
        self.agent = Agent(
            name="Assistant",
            instructions="You are a concise and helpful assistant.",
            model=self.model
        )
    
    def _initialize_gemini(self):
        """Initialize Gemini client and configuration."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file.")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        
        model = OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=client
        )
        
        config = RunConfig(
            model=model,
            model_provider=client,
            tracing_disabled=True
        )
        
        return model, config
    
    def query(self, prompt: str) -> str:
        """Execute a synchronous query against the Gemini agent."""
        try:
            result = Runner.run_sync(self.agent, prompt, run_config=self.config)
            return result.final_output
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

# Example usage
if __name__ == "__main__":
    agent = GeminiAgent()
    print("\nCALLING AGENT\n")
    response = agent.query("Hello, how are you?")
    print(response)