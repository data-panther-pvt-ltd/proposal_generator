"""
Simple Cost Tracker for OpenAI API Usage
Only tracks total cost per execution - no complex analytics
Supports both direct API calls and SDK agent responses
"""
from pathlib import Path
from typing import Optional, Dict, Any

class SimpleCostTracker:
    """Simple cost tracker that only reports total cost per execution"""

    # OpenAI pricing (per token, converted from per 1M tokens - Latest 2025 rates)
    MODEL_PRICING = {
        # GPT-5 series - New 2025 models
        "gpt-5": {"input": 0.00000125, "output": 0.00001, "cached_input": 0.000000125},          # $1.25/$10.00 per 1M
        "gpt-5-mini": {"input": 0.00000025, "output": 0.000002, "cached_input": 0.000000025},    # $0.25/$2.00 per 1M
        "gpt-5-nano": {"input": 0.00000005, "output": 0.0000004, "cached_input": 0.000000005},   # $0.05/$0.40 per 1M
        "gpt-5-chat-latest": {"input": 0.00000125, "output": 0.00001, "cached_input": 0.000000125}, # $1.25/$10.00 per 1M

        # GPT-4o series - Current flagship models
        "gpt-4o": {"input": 0.0000025, "output": 0.00001, "cached_input": 0.00000125},           # $2.50/$10.00 per 1M
        "gpt-4o-2024-05-13": {"input": 0.000005, "output": 0.000015},                            # $5.00/$15.00 per 1M
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006, "cached_input": 0.000000075}, # $0.15/$0.60 per 1M



        # Legacy GPT-4 models
        "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},                                   # $10.00/$30.00 per 1M
        "gpt-4": {"input": 0.00003, "output": 0.00006},                                         # $30.00/$60.00 per 1M
        "gpt-4-32k": {"input": 0.00006, "output": 0.00012},                                     # $60.00/$120.00 per 1M
        "chatgpt-4o-latest": {"input": 0.000005, "output": 0.000015},                           # $5.00/$15.00 per 1M



        # Embedding models (input only, no output tokens) - Current 2025 pricing
        "text-embedding-3-small": {"input": 0.00000002},                                        # $0.02 per 1M
        "text-embedding-3-large": {"input": 0.00000013},                                        # $0.13 per 1M
        "text-embedding-ada-002": {"input": 0.0000001},                                         # $0.10 per 1M
    }
    
    # Batch API offers 50% discount for async requests
    BATCH_DISCOUNT = 0.5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cost tracker"""
        self.config = config or {}
        self.default_model = self.config.get('openai', {}).get('model', 'gpt-4o')
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self.streaming_calls = 0
        self.batch_calls = 0

    def track_completion(self, response, model: str = None):
        if model is None:
            model = self.default_model

        # Handle SDK agent responses
        if self._is_sdk_response(response):
            self._track_sdk_response(response, model)
            return

        # Handle direct OpenAI API responses
        if hasattr(response, 'usage'):
            usage = response.usage
            rate = self.MODEL_PRICING.get(model)
            
            if not rate:
                # Log warning but don't fail
                print(f"Warning: Pricing not found for model: {model}")
                return
            
            # Handle embeddings separately (input only)
            if "output" not in rate:
                input_tokens = usage.total_tokens
                output_tokens = 0
                input_cost = input_tokens * rate["input"]
                output_cost = 0.0
            else:
                # Normal chat/completion models
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                input_cost = input_tokens * rate["input"]
                output_cost = output_tokens * rate["output"]
            
            # Update token counts
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.api_calls += 1
            
            # Calculate and update cost
            cost = input_cost + output_cost
            self.total_cost += cost

    def _is_sdk_response(self, response) -> bool:
        """Check if response is from SDK agent"""
        # SDK responses typically have these attributes
        return (hasattr(response, 'messages') or
                hasattr(response, 'agent') or
                hasattr(response, 'context_variables') or
                str(type(response)).find('agents') != -1)

    def _track_sdk_response(self, response, model: str):
        """Track cost from SDK agent response"""
        try:
            # Try to extract usage from SDK response
            usage = None

            # Method 1: Direct usage attribute
            if hasattr(response, 'usage'):
                usage = response.usage

            # Method 2: Check messages for usage
            elif hasattr(response, 'messages') and response.messages:
                for message in response.messages:
                    if hasattr(message, 'usage'):
                        usage = message.usage
                        break

            # Method 3: Check in response metadata
            elif hasattr(response, 'metadata') and isinstance(response.metadata, dict):
                usage = response.metadata.get('usage')

            # Method 4: Check for nested response object
            elif hasattr(response, 'response') and hasattr(response.response, 'usage'):
                usage = response.response.usage
                
            # Method 5: Check for last_response attribute (common in SDK)
            elif hasattr(response, 'last_response') and hasattr(response.last_response, 'usage'):
                usage = response.last_response.usage
                
            # Method 6: Check for run_result or similar
            elif hasattr(response, 'run_result') and hasattr(response.run_result, 'usage'):
                usage = response.run_result.usage
                
            # Method 7: Check RunResult.raw_responses for usage data
            elif hasattr(response, 'raw_responses') and response.raw_responses:
                for raw_response in response.raw_responses:
                    if hasattr(raw_response, 'usage') and raw_response.usage:
                        usage = raw_response.usage
                        print(f"DEBUG: Found usage in raw_responses: {usage}")
                        break
                        
            # Method 8: Check new_items for usage data  
            elif hasattr(response, 'new_items') and response.new_items:
                for item in response.new_items:
                    if hasattr(item, 'usage') and item.usage:
                        usage = item.usage
                        print(f"DEBUG: Found usage in new_items: {usage}")
                        break

            if usage:
                # Try multiple attribute names for token counts
                input_tokens = (getattr(usage, 'prompt_tokens', None) or 
                              getattr(usage, 'input_tokens', None) or 0)
                output_tokens = (getattr(usage, 'completion_tokens', None) or 
                               getattr(usage, 'output_tokens', None) or 0)
                
                # Debug logging
                print(f"DEBUG: Found SDK usage - Input: {input_tokens}, Output: {output_tokens}, Model: {model}")

                # Update token counts
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.api_calls += 1

                # Calculate cost using exact token pricing
                if model in self.MODEL_PRICING:
                    rate = self.MODEL_PRICING[model]
                    
                    # Handle embeddings (input only)
                    if "output" not in rate:
                        input_cost = input_tokens * rate["input"]
                        output_cost = 0.0
                    else:
                        input_cost = input_tokens * rate["input"]
                        output_cost = output_tokens * rate["output"]
                    
                    cost = input_cost + output_cost
                    self.total_cost += cost
            else:
                # If we can't extract usage, estimate based on content length
                print(f"DEBUG: No usage found in SDK response, falling back to estimation for model: {model}")
                self._estimate_sdk_cost(response, model)

        except Exception as e:
            # Fallback to estimation if usage extraction fails
            self._estimate_sdk_cost(response, model)

    def _estimate_sdk_cost(self, response, model: str):
        """Estimate cost when usage data is not available"""
        # Rough estimation: 1 token ≈ 4 characters
        content_length = 0

        try:
            if hasattr(response, 'messages') and response.messages:
                for message in response.messages:
                    if hasattr(message, 'content'):
                        content_length += len(str(message.content))
            elif hasattr(response, 'content'):
                content_length += len(str(response.content))
            else:
                content_length = len(str(response))

            # Estimate tokens (rough approximation)
            estimated_tokens = content_length // 4
            estimated_input = estimated_tokens // 3  # Assume 1/3 input, 2/3 output
            estimated_output = estimated_tokens - estimated_input

            # Update counts
            self.total_input_tokens += estimated_input
            self.total_output_tokens += estimated_output
            self.api_calls += 1

            # Calculate estimated cost using exact token pricing
            if model in self.MODEL_PRICING:
                rate = self.MODEL_PRICING[model]
                
                # Handle embeddings (input only)
                if "output" not in rate:
                    input_cost = estimated_input * rate["input"]
                    output_cost = 0.0
                else:
                    input_cost = estimated_input * rate["input"]
                    output_cost = estimated_output * rate["output"]
                
                cost = input_cost + output_cost
                self.total_cost += cost

        except Exception:
            # If all else fails, add a minimal default cost
            self.api_calls += 1
            self.total_cost += 0.01  # $0.01 default estimate

    def track_streaming_response(self, stream_chunks, model: str = None):
        if model is None:
            model = self.default_model
        total_input_tokens = 0
        total_output_tokens = 0

        for chunk in stream_chunks:
            if hasattr(chunk, 'usage') and chunk.usage:
                total_input_tokens += getattr(chunk.usage, 'prompt_tokens', 0)
                total_output_tokens += getattr(chunk.usage, 'completion_tokens', 0)

        if total_input_tokens > 0 or total_output_tokens > 0:
            # Update token counts
            self.total_input_tokens += total_input_tokens
            self.total_output_tokens += total_output_tokens
            self.streaming_calls += 1

            # Calculate cost using exact token pricing
            if model in self.MODEL_PRICING:
                rate = self.MODEL_PRICING[model]
                
                # Handle embeddings (input only)
                if "output" not in rate:
                    input_cost = total_input_tokens * rate["input"]
                    output_cost = 0.0
                else:
                    input_cost = total_input_tokens * rate["input"]
                    output_cost = total_output_tokens * rate["output"]
                
                cost = input_cost + output_cost
                self.total_cost += cost

    def get_summary(self) -> dict:
        total_calls = self.api_calls + self.streaming_calls + self.batch_calls
        return {
            'total_cost': round(self.total_cost, 4),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'streaming_calls': self.streaming_calls,
            'batch_calls': self.batch_calls,
            'total_calls': total_calls,
            'average_cost_per_call': round(self.total_cost / max(1, total_calls), 4),
        }

    def track_api_call(self, prompt_tokens: int, completion_tokens: int, model: str = None, section: str = ""):
        if model is None:
            model = self.default_model
        """Track API call with token counts"""
        if model not in self.MODEL_PRICING:
            print(f"Warning: Pricing not found for model: {model}")
            return
            
        rate = self.MODEL_PRICING[model]
        
        # Update token counts
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.api_calls += 1
        
        # Calculate cost using exact token pricing
        if "output" not in rate:
            # Embedding model (input only)
            input_cost = prompt_tokens * rate["input"]
            output_cost = 0.0
        else:
            # Chat/completion model
            input_cost = prompt_tokens * rate["input"]
            output_cost = completion_tokens * rate["output"]
        
        cost = input_cost + output_cost
        self.total_cost += cost

    def track_sdk_completion(self, response, model: str = None):
        if model is None:
            model = self.default_model
        self.track_completion(response, model)

    def track_batch_api_call(self, prompt_tokens: int, completion_tokens: int, model: str = None, section: str = ""):
        if model is None:
            model = self.default_model
        """Track Batch API call with 50% discount"""
        if model not in self.MODEL_PRICING:
            print(f"Warning: Pricing not found for model: {model}")
            return
            
        rate = self.MODEL_PRICING[model]
        
        # Update token counts
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.batch_calls += 1
        
        # Calculate cost using exact token pricing with batch discount
        if "output" not in rate:
            # Embedding model (input only)
            input_cost = prompt_tokens * rate["input"]
            output_cost = 0.0
        else:
            # Chat/completion model
            input_cost = prompt_tokens * rate["input"]
            output_cost = completion_tokens * rate["output"]
        
        # Apply batch discount (50% off)
        cost = (input_cost + output_cost) * self.BATCH_DISCOUNT
        self.total_cost += cost

    def calculate_cost_estimate(self, prompt_tokens: int, completion_tokens: int, model: str = None, use_batch: bool = False) -> dict:
        if model is None:
            model = self.default_model
        """Calculate cost estimate without tracking"""
        if model not in self.MODEL_PRICING:
            return {"error": f"Pricing not found for model: {model}"}
            
        rate = self.MODEL_PRICING[model]
        
        # Calculate base cost
        if "output" not in rate:
            # Embedding model (input only)
            input_cost = prompt_tokens * rate["input"]
            output_cost = 0.0
        else:
            # Chat/completion model
            input_cost = prompt_tokens * rate["input"]
            output_cost = completion_tokens * rate["output"]
        
        base_cost = input_cost + output_cost
        
        # Apply batch discount if requested
        final_cost = base_cost * self.BATCH_DISCOUNT if use_batch else base_cost
        
        return {
            "model": model,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "base_cost": round(base_cost, 6),
            "batch_discount": self.BATCH_DISCOUNT if use_batch else 0,
            "final_cost": round(final_cost, 6),
            "cost_per_1m_input": rate["input"] * 1000000,
            "cost_per_1m_output": rate.get("output", 0) * 1000000 if "output" in rate else None
        }

    def get_total_cost(self) -> float:
        """Get just the total cost"""
        return round(self.total_cost, 4)

    def append_to_cost_md(self, operation_description: str, model_used: str = None, cost_md_path: str = "cost.md") -> bool:
        """
        Append current session costs to cost.md file in proper table format

        Args:
            operation_description: Description of the operation (e.g., "Proposal Generation", "Correction Phase")
            model_used: Primary model used for the operation
            cost_md_path: Path to the cost.md file

        Returns:
            bool: True if successful, False otherwise
        """
        if model_used is None:
            model_used = self.default_model
        import os
        from datetime import datetime
        from pathlib import Path

        try:
            cost_file_path = Path(cost_md_path)

            # Create cost.md if it doesn't exist or is empty
            if not cost_file_path.exists():
                self._create_initial_cost_md(cost_file_path)

            # Read current content
            with open(cost_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # If file exists but is empty, create initial structure
            if not content.strip():
                self._create_initial_cost_md(cost_file_path)
                with open(cost_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Get current cost summary
            summary = self.get_summary()
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            # Skip if no costs to track (avoid writing zero entries)
            if summary['total_cost'] == 0.0 and summary['total_input_tokens'] == 0:
                print(f"⚠️  No costs to track for {operation_description} - cost tracker shows zero usage")
                return False

            # Create new entry for the main table
            new_entry = (
                f"| {current_date} {current_time} | {model_used} | {operation_description} | "
                f"{summary['total_input_tokens']:,} | {summary['total_output_tokens']:,} | "
                f"${summary['total_cost']:.4f} | {operation_description} |\n"
            )

            # Find the main table and insert the new entry
            lines = content.split('\n')
            table_header_found = False
            insert_index = -1

            for i, line in enumerate(lines):
                if '| Date | Model | Operation Type |' in line:
                    table_header_found = True
                    # Find the line after the separator (---|---|---...)
                    if i + 2 < len(lines):
                        insert_index = i + 2
                    break

            if table_header_found and insert_index > 0:
                # Insert the new entry at the beginning of the table data
                lines.insert(insert_index, new_entry.rstrip())

                # Update total costs summary
                updated_content = self._update_total_costs_summary('\n'.join(lines), model_used, summary['total_cost'])

                # Write updated content back to file
                with open(cost_file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

                print(f"✅ Cost tracking updated: ${summary['total_cost']:.4f} for {operation_description}")
                return True
            else:
                print("⚠️  Could not find cost table structure in cost.md")
                return False

        except Exception as e:
            print(f"❌ Failed to append cost to cost.md: {str(e)}")
            return False

    def _create_initial_cost_md(self, cost_file_path: Path):
        """Create initial cost.md file with proper structure"""
        initial_content = """# AI Proposal Generator - Cost Tracking

## Cost Breakdown by Model and Usage

| Date | Model | Operation Type | Input Tokens | Output Tokens | Total Cost (USD) | Description |
|------|-------|----------------|--------------|---------------|------------------|-------------|


"""

        with open(cost_file_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)

    def _update_total_costs_summary(self, content: str, model_used: str, session_cost: float) -> str:
        """Update the total costs summary table"""
        from datetime import datetime
        lines = content.split('\n')
        grand_total = 0.0
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Find and update the summary table
        in_summary_table = False
        summary_table_start = -1

        for i, line in enumerate(lines):
            if '| Model | Total Usage (USD) | Last Updated |' in line:
                in_summary_table = True
                summary_table_start = i
                continue

            if in_summary_table and line.startswith('|') and model_used.upper() in line.upper():
                # Extract current cost and add session cost
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        current_cost_str = parts[2].strip().replace('$', '')
                        current_cost = float(current_cost_str)
                        new_total = current_cost + session_cost

                        # Update the line
                        parts[2] = f" ${new_total:.4f} "
                        parts[3] = f" {current_date} "
                        lines[i] = '|'.join(parts)

                    except (ValueError, IndexError):
                        pass

            if in_summary_table and line.startswith('**Grand Total:'):
                # Calculate new grand total from all model costs
                grand_total = self._calculate_grand_total_from_content('\n'.join(lines))
                lines[i] = f"**Grand Total: ${grand_total:.4f}**"
                break

        return '\n'.join(lines)

    def _calculate_grand_total_from_content(self, content: str) -> float:
        """Calculate grand total from all model costs in the summary table"""
        lines = content.split('\n')
        total = 0.0

        in_summary_table = False
        for line in lines:
            if '| Model | Total Usage (USD) | Last Updated |' in line:
                in_summary_table = True
                continue

            if in_summary_table and line.startswith('|') and '$' in line:
                if 'Model' not in line and 'Total Usage' not in line:  # Skip header
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            cost_str = parts[2].strip().replace('$', '')
                            cost = float(cost_str)
                            total += cost
                        except (ValueError, IndexError):
                            continue

            if in_summary_table and line.startswith('**Grand Total:'):
                break

        return total
