Refactor @backend/ai_generator.py to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds.

Current behaviour:
- Claude makes one tool call -> tool are removed from API params -> final response
- If Claude wishes to make another tool call after seeing the results, it can't (gets empty response)

Desired behaviour:
- Each tool call should be a separate API request so Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when information from different courses/lessons is needed.

Example workflow:
1. User: "Search for a course that discusses the same topic as lesson Y of course X"
2. Claude: gets the course outline for course X, extracts the title of lesson Y
3. Claude: uses the title to search for a course that discusses the same topic, returns course information.
4. Claude: generates the complete answer

Requirements:
- Maximum two sequential rounds per user query
- Terminate when: (a) 2 rounds complete; (b) Claude's response has no tool_use blocks; or (c) tool call fails
- Preserve conversation context between rounds
- Handle tool execution errors gracefully

Notes:
- Update the system prompt in @backend/ai_generator.py
- Update the test scripts
- Ensure the tests verify the external behaviour (API calls made, tool executed, results returned) rather than internal state details
- Update CLAUDE.md

* Use two parallel subagents to brainstorm possible plans. Do not implement any code.
