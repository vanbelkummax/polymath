"""
Polymath Business Intelligence MCP Servers

This module provides MCP servers for business, legal, and financial intelligence:

- patent_mcp: USPTO PatentsView API for patent search and IP landscape analysis
- legal_mcp: CourtListener API for case law search and precedent discovery
- corporate_mcp: OpenCorporates API for company data and corporate structure
- market_intelligence_mcp: GDELT API for news monitoring and sentiment analysis

All servers follow the Polymath MCP pattern with:
- Async operation
- Rate limiting
- Local caching
- Error handling

Environment Variables:
- COURTLISTENER_API_KEY: Optional, increases rate limits
- OPENCORPORATES_API_KEY: Optional, required for some features

Usage:
    # Add to ~/.mcp.json
    "patent-intelligence": {
        "command": "python3",
        "args": ["/home/user/polymath-repo/mcp/business/patent_mcp.py"]
    }
"""

__version__ = "1.0.0"
__author__ = "Polymath"
