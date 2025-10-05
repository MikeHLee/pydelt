#!/bin/bash

# Compile the LaTeX file to PDF
cd /Users/Michaellee/Documents/Runes/pydelt/local/comparisons
pdflatex paper_latex.tex
pdflatex paper_latex.tex  # Run twice for references

echo "Compilation complete. PDF file is at: /Users/Michaellee/Documents/Runes/pydelt/local/comparisons/paper_latex.pdf"
