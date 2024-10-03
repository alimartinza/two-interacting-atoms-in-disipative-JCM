# LaTeX Makefile for automated compilation.
#     USAGE: (a) Run 'make' on the command line to create pdf document.
#            (b) Run 'make clean' to delete PDF document. 
#            (c) Run 'make view' to open the PDF document. 
#            (d) Run 'make zip' to zip all base files including PDF (if it
#                exists).
#            (e) Run 'make edit' to edit the TeX file in vim.
#            (f) Help is now available.
#
#     NOTES: 
#     	(a) latexmk is required.
#     	(b) creates a local folder named 'auxfiles' (if it doesn't exist
#     	    already) for storing auxiliary files.
#
#
#     Pablo Cobelli - 2015

target?=tesis.tex
TEXFILE=$(basename $(target))

$(TEXFILE).pdf: $(TEXFILE).tex
	@rm -f $(TEXFILE).pdf
	@mkdir -p auxfiles
	latexmk -pdf -jobname=auxfiles/$(TEXFILE) $(TEXFILE).tex 
	@mv auxfiles/$(TEXFILE).pdf .
	@echo "Makefile: Auxiliary files available in 'auxfiles' folder."
	@echo "Makefile: Output PDF file '$(TEXFILE).pdf' available in current folder."

.PHONY: clean all

clean: 
	@rm -rf auxfiles 
	@rm -f $(TEXFILE).zip
	@rm -f $(TEXFILE).pdf

view:
	@open $(TEXFILE).pdf

zip:
	@rm -f $(TEXFILE).zip
	@echo "Makefile: zipping base files into '$(TEXFILE).zip'."
	@zip -r $(TEXFILE).zip * --exclude 'auxfiles/*'

edit:
	@vim $(TEXFILE).tex

help:
	@echo ""
	@echo "USAGE: "
	@echo ""
	@echo "  make target=Name_of_tex_file_to_process"
	@echo ""
	@echo "MORE COMMANDS:"
	@echo ""
	@echo "make             - process 'tesis.tex' file (default)"
	@echo "make clean       - cleans all auxiliary files"
	@echo "make edit        - edit the source file in vim"
	@echo "make view        - opens the pdf file in preview"
	@echo "make zip         - zips the contents of current folder"
	@echo ""
	@echo "make help        - displays this help"
	@echo ""
