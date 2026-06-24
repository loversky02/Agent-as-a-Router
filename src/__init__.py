"""Agent-as-a-Router (ACRouter) — a learning router for coding models.

Routing is framed as a Context -> Action -> Feedback -> Context loop:
  featurize task  ->  pick model  ->  score outcome  ->  update experience.
See README.md for how each module maps onto the paper.
"""
