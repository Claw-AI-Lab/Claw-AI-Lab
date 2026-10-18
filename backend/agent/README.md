# ResearchClaw backend

The research runtime for [Claw AI Lab](https://github.com/Claw-AI-Lab/Claw-AI-Lab).
See the repository README for installation, configuration and running the dashboard.

From this directory, `pip install -e ".[parallel]"` installs the optional Parallel
Search MCP adapter. Set `web_search.provider: parallel` in your project YAML to
use free anonymous web search. The default `auto` provider is unchanged.
