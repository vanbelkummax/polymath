# Polymath Knowledge Base - Makefile
# Common operations for knowledge base management

.PHONY: help gemini_passage_pilot gemini_passage_backfill gemini_backfill_status gemini_cost_estimate

PYTHON := python3
SCRIPTS := scripts

help:
	@echo "Polymath Knowledge Base - Available Targets"
	@echo ""
	@echo "Gemini Batch Backfill:"
	@echo "  make gemini_passage_pilot      - Run pilot (200 passages)"
	@echo "  make gemini_passage_backfill   - Run full passage backfill"
	@echo "  make gemini_backfill_status    - Show backfill status"
	@echo "  make gemini_cost_estimate      - Show cost estimate"
	@echo ""
	@echo "Prerequisites:"
	@echo "  export GEMINI_API_KEY=your_api_key"
	@echo ""

# Gemini Batch Backfill Targets
gemini_passage_pilot:
	@echo "Running Gemini passage pilot (200 passages)..."
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "ERROR: GEMINI_API_KEY not set"; \
		echo "Get your key at: https://aistudio.google.com/apikey"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS)/backfill_passage_concepts_gemini_batch.py --pilot
	@echo ""
	@echo "Pilot complete. Review QC report in docs/runlogs/"

gemini_passage_backfill:
	@echo "Running FULL Gemini passage backfill..."
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "ERROR: GEMINI_API_KEY not set"; \
		exit 1; \
	fi
	@echo "This will process ~538k passages. Cost estimate: ~$$10-15"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) $(SCRIPTS)/backfill_passage_concepts_gemini_batch.py --full

gemini_backfill_status:
	@echo "Gemini Backfill Status:"
	@$(PYTHON) $(SCRIPTS)/backfill_passage_concepts_gemini_batch.py --status

gemini_cost_estimate:
	@echo "Estimating cost for remaining passages..."
	@$(PYTHON) $(SCRIPTS)/backfill_passage_concepts_gemini_batch.py --estimate

# Convenience aliases
pilot: gemini_passage_pilot
backfill: gemini_passage_backfill
status: gemini_backfill_status
