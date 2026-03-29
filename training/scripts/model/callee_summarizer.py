# training/scripts/model/callee_summarizer.py
#
# Inter-procedural callee context for Config E ablation.
# Generates CodeBERT [CLS] embeddings for callee functions and maps them
# to CALL nodes in the CPG graph.
#
# Research basis: VulnSC / Inter-proc (ICSE 2024) — callee summary injection.
#
# Usage:
#   summarizer = CalleeSummarizer(cache=CalleeCache())
#   context = summarizer.get_callee_context(cpg_json, callee_sources)
#   # context = [(node_idx, 768-d tensor), ...]
#   # Pass to model.forward(callee_node_indices=..., callee_embeddings=...)

import logging

import torch
from transformers import AutoModel, AutoTokenizer

from training.scripts.model.callee_cache import CalleeCache

logger = logging.getLogger(__name__)

CODEBERT_DIM = 768


class CalleeSummarizer:
    """
    Generates CodeBERT embeddings for callee functions and maps them
    to CALL nodes in the CPG for inter-procedural context injection.

    Used only for ablation Config E (use_interproc=True).
    Configs A-D do not instantiate this class.
    """

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        cache: CalleeCache | None = None,
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)
        self.model = AutoModel.from_pretrained(codebert_model).to(device).eval()
        self.cache = cache if cache is not None else CalleeCache()
        self.device = device
        self._call_count = 0  # tracks CodeBERT forward calls (for testing/diagnostics)

    @torch.no_grad()
    def embed_callee(self, callee_code: str) -> torch.Tensor:
        """
        Compute CodeBERT [CLS] embedding for a callee function body.

        Returns 768-d tensor on self.device. Uses cache to avoid
        redundant CodeBERT forward passes.
        """
        # Cache lookup
        cached = self.cache.get(callee_code)
        if cached is not None:
            return cached.to(self.device)

        # Tokenize and encode
        inputs = self.tokenizer(
            callee_code,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs)
        embed = out.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
        self._call_count += 1

        # Cache the result (store on CPU to save GPU memory)
        self.cache.set(callee_code, embed.cpu())
        return embed

    def get_callee_context(
        self,
        cpg_json: dict,
        callee_sources: dict[str, str],
    ) -> list[tuple[int, torch.Tensor]]:
        """
        Extract callee embeddings for CALL nodes in a CPG.

        For each node in cpg_json["nodes"]:
          - Skip if node["_label"] != "CALL"
          - Skip if callee function name not in callee_sources
          - Otherwise embed the callee body and return (node_idx, embed_768d)

        Args:
            cpg_json: CPG dict with "nodes" list. Each node has "_label" and "name".
            callee_sources: {function_name: source_code} for known callees.

        Returns:
            List of (local_node_idx, 768-d tensor) tuples.
            Empty list if no CALL nodes match known callees.
        """
        if not callee_sources:
            return []

        results = []
        nodes = cpg_json.get("nodes", [])

        for node_idx, node in enumerate(nodes):
            # Only process CALL nodes
            if node.get("_label") != "CALL":
                continue

            # Extract function name (strip namespace/qualifier if present)
            raw_name = node.get("name", "")
            fname = raw_name.split(".")[-1] if "." in raw_name else raw_name

            if not fname or fname not in callee_sources:
                continue

            try:
                embed = self.embed_callee(callee_sources[fname])
                results.append((node_idx, embed))
            except Exception as e:
                logger.debug("Callee embed failed for %s: %s", fname, e)

        return results

    def prepare_batch_callee_context(
        self,
        batch_cpg_jsons: list[dict],
        batch_callee_sources: list[dict[str, str]],
    ) -> list[tuple[int, int, torch.Tensor]]:
        """
        Process a full batch of CPGs for inter-proc injection.

        Returns list of (graph_idx, local_node_idx, callee_embed_768d) tuples
        suitable for model.forward(callee_node_indices=...).
        """
        all_tuples = []
        for graph_idx, (cpg_json, callee_sources) in enumerate(
            zip(batch_cpg_jsons, batch_callee_sources)
        ):
            if callee_sources is None:
                continue
            node_embeds = self.get_callee_context(cpg_json, callee_sources)
            for node_idx, embed in node_embeds:
                all_tuples.append((graph_idx, node_idx, embed))
        return all_tuples
