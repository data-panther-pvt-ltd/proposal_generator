"""
Simple Cost Tracker for OpenAI API Usage
Only tracks total cost per execution - no complex analytics
Supports both direct API calls and SDK agent responses
"""

class SimpleCostTracker:
    """Simple cost tracker that only reports total cost per execution"""

    # OpenAI pricing as of 2025 (per 1M tokens)
    MODEL_PRICING = {
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4o-2024-05-13': {'input': 5.00, 'output': 15.00},
        'gpt-4o-audio-preview': {'input': 2.50, 'output': 10.00},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        'gpt-3.5-turbo-0125': {'input': 0.50, 'output': 1.50},
        'text-embedding-3-small': {'input': 0.01, 'output': 0.00},
        'text-embedding-3-large': {'input': 0.065, 'output': 0.00},
        'text-embedding-ada-002': {'input': 0.05, 'output': 0.00}
    }

    def __init__(self):
        """Initialize the cost tracker"""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self.streaming_calls = 0

    def track_completion(self, response, model: str = "gpt-4o"):

        # Handle SDK agent responses
        if self._is_sdk_response(response):
            self._track_sdk_response(response, model)
            return

        # Handle direct OpenAI API responses
        if hasattr(response, 'usage'):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Update token counts
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.api_calls += 1

            # Calculate cost (pricing is per 1M tokens)
            if model in self.MODEL_PRICING:
                pricing = self.MODEL_PRICING[model]
                input_cost = (input_tokens / 1_000_000) * pricing['input']
                output_cost = (output_tokens / 1_000_000) * pricing['output']
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

            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)

                # Update token counts
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.api_calls += 1

                # Calculate cost (pricing is per 1M tokens)
                if model in self.MODEL_PRICING:
                    pricing = self.MODEL_PRICING[model]
                    input_cost = (input_tokens / 1_000_000) * pricing['input']
                    output_cost = (output_tokens / 1_000_000) * pricing['output']
                    cost = input_cost + output_cost
                    self.total_cost += cost
            else:
                # If we can't extract usage, estimate based on content length
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

            # Calculate estimated cost (pricing is per 1M tokens)
            if model in self.MODEL_PRICING:
                pricing = self.MODEL_PRICING[model]
                input_cost = (estimated_input / 1_000_000) * pricing['input']
                output_cost = (estimated_output / 1_000_000) * pricing['output']
                cost = input_cost + output_cost
                self.total_cost += cost

        except Exception:
            # If all else fails, add a minimal default cost
            self.api_calls += 1
            self.total_cost += 0.01  # $0.01 default estimate

    def track_streaming_response(self, stream_chunks, model: str = "gpt-4o"):
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

            # Calculate cost (pricing is per 1M tokens)
            if model in self.MODEL_PRICING:
                pricing = self.MODEL_PRICING[model]
                input_cost = (total_input_tokens / 1_000_000) * pricing['input']
                output_cost = (total_output_tokens / 1_000_000) * pricing['output']
                cost = input_cost + output_cost
                self.total_cost += cost

    def get_summary(self) -> dict:
        return {
            'total_cost': round(self.total_cost, 4),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'streaming_calls': self.streaming_calls,
            'total_calls': self.api_calls + self.streaming_calls,
            'average_cost_per_call': round(self.total_cost / max(1, self.api_calls + self.streaming_calls), 4),
        }

    def track_sdk_completion(self, response, model: str = "gpt-4o"):
        self.track_completion(response, model)

    def get_total_cost(self) -> float:
        """Get just the total cost"""
        return round(self.total_cost, 4)
