#!/bin/bash

# Script de conversion Markdown vers PDF avec Pandoc

pandoc README.md -o Rapport_Regression_Lineaire.pdf \
    --template=template/template.tex \
    --toc \
    --toc-depth=2 \
    --listings

echo "PDF généré : Rapport_Regression_Lineaire.pdf"
