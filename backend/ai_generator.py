import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Maximum sequential tool-calling rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search course materials for specific content or educational details
2. **get_course_outline**: Get course structure including title, link, and complete lesson list with links

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content, concepts, or detailed material
- Use **get_course_outline** for questions about course structure, lesson lists, what topics a course covers, or course outlines
- **Up to 2 sequential tool rounds available** - Use multiple rounds when one tool's results inform the next search (e.g., get course outline first, then search based on lesson title)
- Synthesize all tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **Multi-step queries**: Chain tool calls when needed (e.g., first get outline to find lesson title, then search for related content)
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "based on the outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool-calling rounds, allowing
        Claude to make multiple tool calls where each round's results can inform
        the next tool call.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message list with user query
        messages = [{"role": "user", "content": query}]

        # Track rounds
        round_count = 0

        # Main loop for sequential tool calling
        while round_count < self.MAX_TOOL_ROUNDS:
            # Build API parameters - include tools if available
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self.client.messages.create(**api_params)

            # Check stop reason
            if response.stop_reason in ("end_turn", "max_tokens"):
                # Claude is done - extract and return text
                return self._extract_text_response(response)

            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and prepare for next round
                tool_results, has_error = self._execute_tools(response, tool_manager)

                # Append assistant's tool_use response
                messages.append({"role": "assistant", "content": response.content})

                # Append tool results as user message
                messages.append({"role": "user", "content": tool_results})

                if has_error:
                    # Critical tool failure - make final call without tools
                    return self._final_call_without_tools(messages, system_content)

                round_count += 1
                continue

            # Unexpected stop reason - return whatever text we have
            return self._extract_text_response(response)

        # Loop exhausted (MAX_TOOL_ROUNDS reached)
        # Make final call WITHOUT tools to get synthesis
        return self._final_call_without_tools(messages, system_content)

    def _execute_tools(self, response, tool_manager) -> tuple:
        """
        Execute all tool calls from a response.

        Args:
            response: The API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, has_critical_error bool)
        """
        tool_results = []
        has_critical_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })

                except Exception as e:
                    # Tool execution failed
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution error: {str(e)}",
                        "is_error": True
                    })
                    has_critical_error = True

        return tool_results, has_critical_error

    def _extract_text_response(self, response) -> str:
        """
        Extract text from response, handling mixed content blocks.

        A response may contain both text and tool_use blocks.
        This extracts only the text portions.

        Args:
            response: The API response

        Returns:
            Combined text content from all text blocks
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)

        return "\n".join(text_parts) if text_parts else ""

    def _final_call_without_tools(self, messages: List, system_content: str) -> str:
        """
        Make final API call without tools to synthesize response.

        Used when max rounds are reached or after tool errors.

        Args:
            messages: Accumulated conversation messages
            system_content: System prompt content

        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        response = self.client.messages.create(**final_params)
        return self._extract_text_response(response)