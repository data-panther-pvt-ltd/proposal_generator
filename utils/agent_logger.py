"""
Agent-specific logging configuration
Creates individual log files for each agent
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import yaml

class AgentLogger:
    """Configure agent-specific logging"""

    def __init__(self, log_dir: str = "logs/agents"):
        self.log_dir = Path(log_dir)
        self.loggers = {}
        self.enabled = self._check_logging_enabled()

        # Only create directories if logging is enabled
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _check_logging_enabled(self):
        """Check if agent logging is enabled in config"""
        try:
            with open('config/settings.yml', 'r') as f:
                config = yaml.safe_load(f)
            return config.get('logging', {}).get('save_agent_interactions', False)
        except:
            return False
        
    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get or create a logger for a specific agent"""

        if agent_name in self.loggers:
            return self.loggers[agent_name]

        # Create agent-specific logger
        logger = logging.getLogger(f"agent.{agent_name}")

        # If logging is disabled, return a simple logger with no handlers
        if not self.enabled:
            logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
            logger.handlers = []
            self.loggers[agent_name] = logger
            return logger

        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers = []

        # Create file handler for this agent
        log_file = self.log_dir / f"{agent_name}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        # Log initialization
        logger.info(f"{'='*60}")
        logger.info(f"Agent Logger Initialized: {agent_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"{'='*60}")

        # Store logger
        self.loggers[agent_name] = logger

        return logger
    
    def log_agent_execution(self, agent_name: str, task: str, context: dict = None, result: dict = None):
        """Log detailed agent execution information"""
        if not self.enabled:
            return

        if context is None:
            context = {}
        if result is None:
            result = {}
        logger = self.get_agent_logger(agent_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT EXECUTION START: {agent_name}")
        logger.info(f"Task: {task}")
        logger.info(f"Context Keys: {list(context.keys())}")

        # Log request details if available
        if 'request' in context:
            request = context['request']
            if hasattr(request, '__dict__'):
                logger.info(f"Request Details:")
                for key, value in request.__dict__.items():
                    if not key.startswith('_'):
                        logger.info(f"  - {key}: {str(value)[:100]}")

        # Log result
        logger.info(f"Result Type: {type(result)}")
        if isinstance(result, dict):
            logger.info(f"Result Keys: {list(result.keys())}")
            if 'error' in result:
                logger.error(f"EXECUTION FAILED: {result['error']}")
            else:
                logger.info(f"EXECUTION SUCCESS")
                if 'content' in result:
                    logger.info(f"Content Length: {len(str(result['content']))}")

        logger.info(f"AGENT EXECUTION END: {agent_name}")
        logger.info(f"{'='*60}\n")

    def log_tool_call(self, agent_name: str, tool_name: str, params: dict, result: any):
        """Log tool usage by agents"""
        if not self.enabled:
            return

        logger = self.get_agent_logger(agent_name)

        logger.info(f"TOOL CALL: {tool_name}")
        logger.info(f"Parameters: {params}")
        logger.info(f"Result Type: {type(result)}")
        if result:
            logger.info(f"Result: {str(result)[:200]}")

# Global agent logger instance
agent_logger = AgentLogger()

# Pre-initialize loggers for known agents (only if logging is enabled)
if agent_logger.enabled:
    for agent in ['coordinator', 'content_generator', 'researcher', 'budget_calculator', 'quality_evaluator']:
        agent_logger.get_agent_logger(agent)