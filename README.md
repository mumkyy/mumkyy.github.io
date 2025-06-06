# Optical Core ViT Analyzer

This repository hosts a small, single page tool that explores how Vision Transformer parameters map onto a photonic optical core.  It is designed for GitHub Pages—open `index.html` directly or host it from your repository.

Parameters such as image size, patch size, embedding dimension, attention dimension, number of heads, and core properties can be adjusted.  The tool computes estimated optical core cycle counts for the patch‑embedding, attention, and weighted‑sum phases and plots them using a bar chart.

A collapsible **Math Explanation** section on the page describes the formulas used so you can verify the calculations yourself.

No build step or external dependencies are required—everything loads from CDNs.
