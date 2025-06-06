# Optical Core ViT Analyzer

This repository contains a lightweight, GitHub Pages compatible tool for exploring how Vision Transformer (ViT) parameters map onto a photonic optical core.

Open `index.html` in a browser (or via GitHub Pages) to adjust parameters such as image size, patch size, embedding dimension, attention dimension, number of heads, and core properties. The page will display approximate optical core cycle counts for patch embedding, attention, and weighted-sum phases.

The calculations follow the logic outlined in the accompanying problem description. They assume that each optical core processes inputs in groups equal to the available wavelength channels.
