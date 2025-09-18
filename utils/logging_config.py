"""
Centralized logging configuration for the Proposal Generator
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log output"""
    
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BOLD + Colors.RED,
    }
    
    AGENT_COLORS = {
        'coordinator': Colors.BLUE,
        'content_generator': Colors.GREEN,
        'researcher': Colors.MAGENTA,
        'budget_calculator': Colors.YELLOW,
        'quality_evaluator': Colors.CYAN        
    }
    
    def format(self, record):
        # Add color based on log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"
        
        # Add agent-specific coloring if available
        if hasattr(record, 'agent'):
            agent = record.agent
            if agent in self.AGENT_COLORS:
                record.msg = f"{self.AGENT_COLORS[agent]}[{agent.upper()}]{Colors.RESET} {record.msg}"
        
        return super().format(record)

class AgentLogHandler(logging.Handler):
    """Custom handler that saves agent interactions to separate files"""
    
    def __init__(self, logs_dir: str):
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.agent_files = {}
        
    def emit(self, record):
        try:
            if hasattr(record, 'agent'):
                agent = record.agent
                
                # Create agent-specific log file if not exists
                if agent not in self.agent_files:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_file = self.logs_dir / f"{agent}_{timestamp}.json"
                    self.agent_files[agent] = log_file
                
                # Save interaction data
                interaction = {
                    'timestamp': datetime.now().isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                }
                
                # Add extra data if available
                if hasattr(record, 'prompt'):
                    interaction['prompt'] = record.prompt
                if hasattr(record, 'response'):
                    interaction['response'] = record.response
                if hasattr(record, 'tokens'):
                    interaction['tokens'] = record.tokens
                if hasattr(record, 'cost'):
                    interaction['cost'] = record.cost
                if hasattr(record, 'error'):
                    interaction['error'] = record.error
                
                # Append to JSON file
                with open(self.agent_files[agent], 'a') as f:
                    json.dump(interaction, f)
                    f.write('\n')
                    
        except Exception as e:
            self.handleError(record)

class ProposalLogger:
    """Main logger class for the proposal generator"""
    
    def __init__(self, config: Dict):
        self.config = config['logging']
        self.setup_logging()
        
    def setup_logging(self):
        """Setup all logging configurations"""

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = getattr(logging, self.config.get('console_level', 'INFO'))
        console_handler.setLevel(console_level)

        if self.config.get('colored_output', True):
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        console_handler.setFormatter(console_formatter)

        # Add console handler
        root_logger.addHandler(console_handler)

        # File handler for main log (only if file logging is enabled)
        if self.config.get('file'):
            # Create logs directory only if needed
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(self.config['file'])
            file_level = getattr(logging, self.config.get('level', 'INFO'))
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(self.config['format'])
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Agent interaction handler (only if enabled)
        if self.config.get('save_agent_interactions', False) and self.config.get('agent_logs_dir'):
            agent_handler = AgentLogHandler(self.config.get('agent_logs_dir', 'logs/agents'))
            agent_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(agent_handler)
        
        # Silence verbose PDF-related loggers
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('pdfminer.psparser').setLevel(logging.WARNING)
        logging.getLogger('pdfminer.pdfinterp').setLevel(logging.WARNING)
        logging.getLogger('pdfminer.converter').setLevel(logging.WARNING)
        logging.getLogger('pdfminer.pdfpage').setLevel(logging.WARNING)
        logging.getLogger('pdfminer.pdfdocument').setLevel(logging.WARNING)
        logging.getLogger('pdfplumber').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('openai._base_client').setLevel(logging.WARNING)
        
        # Create agent-specific loggers
        self.setup_agent_loggers()
        
    def setup_agent_loggers(self):
        """Setup individual loggers for each agent (only if agent logging is enabled)"""
        if not self.config.get('save_agent_interactions', False) or not self.config.get('agent_logs_dir'):
            return

        agents = [
            'coordinator',
            'content_generator',
            'researcher',
            'budget_calculator',
            'quality_evaluator'
        ]

        for agent in agents:
            agent_logger = logging.getLogger(f'agent.{agent}')
            agent_logger.setLevel(logging.DEBUG)

            # Create agent-specific file handler
            agent_file = Path('logs') / 'agents' / f'{agent}.log'
            agent_file.parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(agent_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            agent_logger.addHandler(handler)
    
    @staticmethod
    def log_agent_call(agent_name: str, task: str, context: Dict):
        """Log when an agent is called"""
        logger = logging.getLogger(f'agent.{agent_name}')
        logger.info(f"Agent {agent_name} called for task: {task}")
        logger.debug(f"Context keys: {list(context.keys())}")
        
        # Also log to main logger with agent context
        main_logger = logging.getLogger('agent_orchestrator')
        main_logger.info(
            f"Calling agent: {agent_name}",
            extra={'agent': agent_name, 'task': task}
        )
    
    @staticmethod
    def log_agent_response(agent_name: str, response: Dict, tokens: Optional[int] = None):
        """Log agent response"""
        logger = logging.getLogger(f'agent.{agent_name}')
        logger.info(f"Agent {agent_name} responded successfully")
        
        if tokens:
            logger.debug(f"Tokens used: {tokens}")
        
        # Log response keys for debugging
        if isinstance(response, dict):
            logger.debug(f"Response keys: {list(response.keys())}")
        
        # Also log to main logger
        main_logger = logging.getLogger('agent_orchestrator')
        main_logger.info(
            f"Agent response received",
            extra={'agent': agent_name, 'response': response, 'tokens': tokens}
        )
    
    @staticmethod
    def log_agent_error(agent_name: str, error: Exception):
        """Log agent error"""
        logger = logging.getLogger(f'agent.{agent_name}')
        logger.error(f"Agent {agent_name} failed: {str(error)}", exc_info=True)
        
        # Also log to main logger
        main_logger = logging.getLogger('agent_orchestrator')
        main_logger.error(
            f"Agent error",
            extra={'agent': agent_name, 'error': str(error)}
        )
    
    @staticmethod
    def log_api_call(model: str, prompt_tokens: int, completion_tokens: int, cost: float):
        """Log API call details"""
        logger = logging.getLogger('api_calls')
        logger.info(
            f"API Call - Model: {model}, Prompt: {prompt_tokens}, "
            f"Completion: {completion_tokens}, Cost: ${cost:.4f}"
        )

def setup_logging(config: Dict) -> ProposalLogger:
    """Setup logging for the entire application"""
    return ProposalLogger(config)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)