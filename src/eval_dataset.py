"""
Eval dataset for RAGAs evaluation (Phase 0).
Questions cover the core domain of the RAG system (Raman spectroscopy of carbon nanotubes),
mixing factual, comparative, and application-oriented questions to get a representative
picture of retrieval and generation quality — not just easy, single-fact lookups.

No ground truth answers here (Option A) — RAGAs will score faithfulness and context
precision without needing a pre-written reference answer.
"""

EVAL_QUESTIONS = [
    # Basic factual — core Raman bands
    "What is the D band in Raman spectroscopy of carbon nanotubes?",
    "What is the G band and what does it look like?",
    "What is the D/G ratio and why is it important?",

    # RBM — nanotube-specific feature
    "What is the Radial Breathing Mode (RBM) and what does it tell us about a nanotube?",
    "How is the RBM frequency related to nanotube diameter?",

    # Structural / chirality
    "How does chirality affect the Raman spectrum of a carbon nanotube?",
    "What is the difference between single-walled and multi-walled carbon nanotubes in terms of Raman signatures?",

    # Measurement / methodology
    "Why does the laser wavelength affect the position of the D band?",
    "What experimental factors can influence the intensity ratio of D and G bands?",

    # Defects / disorder
    "How can Raman spectroscopy be used to evaluate defects in carbon nanotubes?",
    "What does a high D/G ratio indicate about a carbon nanotube sample?",

    # Comparative / synthesis-related
    "What are the second-order Raman features observed in carbon nanotubes?",
    "How does resonance Raman spectroscopy help characterize carbon nanotube samples?",

    # Slightly out-of-scope (sanity check — should NOT hallucinate)
    "What is the melting point of carbon nanotubes?",
]
