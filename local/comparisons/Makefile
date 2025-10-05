all: paper_pydelt.pdf

paper_pydelt.pdf: paper_pydelt.tex
	pdflatex paper_pydelt.tex
	pdflatex paper_pydelt.tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.nav *.snm *.vrb *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f paper_pydelt.pdf
