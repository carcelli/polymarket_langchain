"""
ML Agent - Intelligent Agent for Automated Machine Learning

An intelligent agent that uses ML tools to perform complete automated
machine learning workflows for Polymarket data analysis and prediction.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from .ml_tools import (
    data_ingestion_tool,
    data_quality_tool,
    automl_pipeline_tool,
    model_training_tool,
    model_evaluation_tool,
    prediction_tool,
    experiment_results_tool,
)
from .ml_database import MLDatabase

logger = logging.getLogger(__name__)


class MLAgent:
    """
    Intelligent agent for automated machine learning workflows.

    Uses LangChain to orchestrate ML operations including:
    - Data ingestion and quality validation
    - Model training and evaluation
    - Prediction and deployment
    - Experiment tracking and reporting
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the ML Agent."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.database = MLDatabase()

        # Set up tools
        self.tools = [
            data_ingestion_tool,
            data_quality_tool,
            automl_pipeline_tool,
            model_training_tool,
            model_evaluation_tool,
            prediction_tool,
            experiment_results_tool,
        ]

        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Create the agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        logger.info("ML Agent initialized with tools and capabilities")

    def _create_agent(self):
        """Create the LangChain agent with ML capabilities."""

        system_prompt = """
        You are an expert ML Engineer specializing in automated machine learning for prediction markets.

        Your capabilities include:
        1. **Data Management**: Ingest, clean, and validate market data from Polymarket
        2. **Model Development**: Train, evaluate, and optimize ML models for market prediction
        3. **Experiment Tracking**: Track experiments, models, and performance metrics
        4. **Prediction Services**: Make predictions on new market data
        5. **Quality Assurance**: Validate data quality and model performance

        **WORKFLOW PRINCIPLES:**
        - Always start with data quality assessment before training
        - Use appropriate evaluation methods (backtesting for trading models)
        - Track all experiments and results in the database
        - Provide clear explanations of ML decisions and results
        - Focus on actionable insights for trading strategies

        **AVAILABLE TOOLS:**
        - data_ingestion: Fetch and prepare historical market data
        - data_quality_check: Validate data quality for ML readiness
        - run_automl_pipeline: Execute complete automated ML pipeline
        - train_ml_model: Train specific ML models
        - evaluate_ml_model: Evaluate trained model performance
        - make_ml_prediction: Generate predictions for market data
        - get_experiment_results: Retrieve experiment and model results

        **BEST PRACTICES:**
        - Use cross-validation for robust model evaluation
        - Focus on F1 score and precision for imbalanced trading data
        - Always validate models on out-of-sample data
        - Document model limitations and assumptions
        - Prioritize models with good calibration for probability predictions

        When asked to perform ML tasks, break them down into logical steps and use the appropriate tools.
        Always provide clear explanations of what you're doing and why.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        return create_openai_functions_agent(
            llm=self.llm, tools=self.tools, prompt=prompt
        )

    def run_ml_workflow(self, task: str) -> Dict[str, Any]:
        """
        Execute an ML workflow task.

        Args:
            task: Description of the ML task to perform

        Returns:
            Results of the ML workflow
        """
        logger.info(f"Starting ML workflow: {task}")

        try:
            # Enhance the task with context about available capabilities
            enhanced_task = f"""
            ML Task: {task}

            Available Capabilities:
            - Data ingestion from Polymarket API
            - Quality validation and cleaning
            - Automated ML pipeline execution
            - Individual model training (MarketPredictor, EdgeDetector)
            - Model evaluation and backtesting
            - Real-time prediction generation
            - Experiment tracking and reporting

            Please execute this task using the appropriate ML tools and provide detailed results.
            """

            result = self.agent_executor.invoke({"input": enhanced_task})

            # Parse and structure the result
            output = result.get("output", "")
            structured_result = self._parse_agent_output(output)

            # Add metadata
            structured_result.update(
                {
                    "task": task,
                    "timestamp": datetime.now().isoformat(),
                    "agent_version": "1.0",
                    "tools_used": [tool.name for tool in self.tools],
                }
            )

            logger.info(f"ML workflow completed: {task}")
            return structured_result

        except Exception as e:
            logger.error(f"ML workflow failed: {e}")
            return {
                "status": "error",
                "task": task,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        """Parse agent output into structured results."""
        try:
            # Try to parse as JSON first
            parsed = json.loads(output)
            return parsed
        except json.JSONDecodeError:
            # If not JSON, extract structured information
            result = {"status": "success", "raw_output": output, "parsed_info": {}}

            # Extract key information from text
            lines = output.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("✅") or line.startswith("❌"):
                    if "success" not in result["parsed_info"]:
                        result["parsed_info"]["success"] = []
                    result["parsed_info"]["success"].append(line)
                elif "model_id" in line.lower() or "experiment_id" in line.lower():
                    # Extract IDs
                    if ":" in line:
                        key, value = line.split(":", 1)
                        result["parsed_info"][
                            key.strip().lower().replace(" ", "_")
                        ] = value.strip()

            return result

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models."""
        try:
            models_df = self.database.get_best_models(limit=20)
            return models_df.to_dict("records")
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ML experiments."""
        # This would query the database for recent experiments
        # Simplified implementation
        return []

    def create_ml_report(self, experiment_ids: List[str] = None) -> str:
        """
        Generate a comprehensive ML report.

        Args:
            experiment_ids: Specific experiments to include (None for all recent)

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# ML Agent Report")
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # Database statistics
        try:
            db_stats = self.database.get_database_stats()
            report_lines.append("## Database Statistics")
            report_lines.append(
                f"- Total Experiments: {db_stats.get('experiments_count', 0)}"
            )
            report_lines.append(f"- Total Models: {db_stats.get('models_count', 0)}")
            report_lines.append(
                f"- Total Predictions: {db_stats.get('predictions_count', 0)}"
            )
            report_lines.append(
                f"- Database Size: {db_stats.get('database_size_bytes', 0) / 1024:.1f} KB"
            )
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"## Database Error: {e}")
            report_lines.append("")

        # Recent experiments
        try:
            # Simplified - would query actual recent experiments
            report_lines.append("## Recent Activity")
            report_lines.append("- Model training and evaluation workflows")
            report_lines.append("- Data quality validation")
            report_lines.append("- Automated pipeline execution")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"## Activity Error: {e}")
            report_lines.append("")

        # Best models
        try:
            best_models = self.database.get_best_models(limit=5)
            if not best_models.empty:
                report_lines.append("## Best Performing Models")
                for _, model in best_models.iterrows():
                    report_lines.append(
                        f"- **{model['name']}**: F1 = {model.get('f1_score', 0):.3f} ({model['training_samples']} samples)"
                    )
                report_lines.append("")
        except Exception as e:
            report_lines.append(f"## Models Error: {e}")
            report_lines.append("")

        # Active alerts
        try:
            alerts = self.database.get_active_alerts()
            if alerts:
                report_lines.append("## Active Alerts")
                for alert in alerts[:5]:
                    report_lines.append(
                        f"- {alert['severity'].upper()}: {alert['message']}"
                    )
                report_lines.append("")
        except Exception as e:
            report_lines.append(f"## Alerts Error: {e}")
            report_lines.append("")

        report_lines.append("## Recommendations")
        report_lines.append("1. Monitor model performance regularly")
        report_lines.append("2. Retrain models with new market data")
        report_lines.append("3. Validate predictions on recent markets")
        report_lines.append("4. Review and resolve active alerts")
        report_lines.append("5. Consider ensemble methods for improved performance")

        return "\n".join(report_lines)

    def optimize_model_hyperparameters(
        self, model_type: str, param_grid: Dict[str, List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model type.

        Args:
            model_type: Type of model to optimize
            param_grid: Parameter grid to search

        Returns:
            Optimization results
        """
        if param_grid is None:
            # Default parameter grids
            if model_type == "MarketPredictor":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                }
            elif model_type == "EdgeDetector":
                param_grid = {
                    "hidden_layers": [[32], [64, 32], [128, 64, 32]],
                    "learning_rate": [0.001, 0.01, 0.1],
                    "dropout_rate": [0.1, 0.2, 0.3],
                }
            else:
                return {"error": f"Unknown model type: {model_type}"}

        # Create optimization task
        task = f"""
        Optimize hyperparameters for {model_type} model.

        Parameter grid: {json.dumps(param_grid, indent=2)}

        Steps:
        1. Train models with different parameter combinations
        2. Evaluate each model using cross-validation
        3. Return the best performing parameter set
        4. Save optimization results to database
        """

        return self.run_ml_workflow(task)

    def validate_model_drift(
        self, model_id: str, new_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Check for model drift using new data.

        Args:
            model_id: ID of model to check
            new_data: New data to validate against

        Returns:
            Drift analysis results
        """
        task = f"""
        Check for model drift in model {model_id}.

        Steps:
        1. Load the specified model
        2. Get recent market data (if not provided)
        3. Make predictions on new data
        4. Compare predictions to expected patterns
        5. Calculate drift metrics (PSI, KS test, etc.)
        6. Generate drift alert if significant drift detected
        """

        return self.run_ml_workflow(task)

    def retrain_model(self, model_id: str, new_data_days: int = 30) -> Dict[str, Any]:
        """
        Retrain a model with new data.

        Args:
            model_id: ID of model to retrain
            new_data_days: Days of new data to include

        Returns:
            Retraining results
        """
        task = f"""
        Retrain model {model_id} with recent data.

        Steps:
        1. Load existing model and its training data
        2. Fetch {new_data_days} days of new market data
        3. Combine old and new training data
        4. Retrain model on combined dataset
        5. Evaluate performance improvement
        6. Save retrained model with new version
        7. Compare old vs new model performance
        """

        return self.run_ml_workflow(task)

    def generate_trading_strategy(
        self, model_id: str, risk_tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate a complete trading strategy using an ML model.

        Args:
            model_id: ID of model to use
            risk_tolerance: Risk tolerance for position sizing

        Returns:
            Trading strategy specification
        """
        task = f"""
        Generate trading strategy using model {model_id}.

        Strategy Parameters:
        - Risk Tolerance: {risk_tolerance}
        - Model: {model_id}

        Steps:
        1. Load and analyze the specified model
        2. Define entry/exit criteria based on model predictions
        3. Set position sizing rules using Kelly Criterion
        4. Define risk management rules (stop losses, etc.)
        5. Backtest strategy on historical data
        6. Generate strategy performance report
        7. Create strategy specification document
        """

        return self.run_ml_workflow(task)


def create_ml_agent(model_name: str = "gpt-4o") -> MLAgent:
    """
    Factory function to create an ML Agent.

    Args:
        model_name: LLM model to use

    Returns:
        Configured ML Agent
    """
    return MLAgent(model_name=model_name)


# Example usage and testing functions
def demo_ml_agent():
    """Demonstrate ML Agent capabilities."""
    print("🤖 ML Agent Demo")
    print("=" * 50)

    agent = create_ml_agent()

    # Example workflows
    workflows = [
        "Ingest the last 180 days of market data and check its quality",
        "Train a MarketPredictor model on recent data",
        "Evaluate the trained model's performance using backtesting",
        "Generate a comprehensive ML report of all experiments",
    ]

    results = []
    for workflow in workflows:
        print(f"\\n🎯 Executing: {workflow}")
        result = agent.run_ml_workflow(workflow)
        results.append(result)

        # Print summary
        if result.get("status") == "success":
            print("✅ Completed successfully")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    print("\\n📊 Demo Results Summary:")
    print(f"- Workflows executed: {len(results)}")
    print(f"- Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"- Failed: {sum(1 for r in results if r.get('status') != 'success')}")

    return results


if __name__ == "__main__":
    # Run demo
    demo_ml_agent()
