# Makefile for the zip distribution.
PYTHON := python3

# List source files here.
MPCTOOLS_SRC := $(addprefix mpctools/, __init__.py colloc.py plots.py \
                  solvers.py tools.py util.py mpcsim.py compat.py)

EXAMPLES := airplane.py ballmaze.py cstr.py cstr_startup.py cstr_nmpc_nmhe.py \
            collocationexample.py comparison_casadi.py comparison_mtc.py \
            econmpc.py example2-8.py mheexample.py mpcexampleclosedloop.py \
            mpcmodelcomparison.py nmheexample.py nmpcexample.py \
            periodicmpcexample.py vdposcillator.py predatorprey.py \
            cargears.py fishing.py runall.py siso_lmpc_mpcsim.py \
            cstr_lqg_mpcsim.py cstr_nmpc_mpcsim.py heater_pid_mpcsim.py \
            template.py icyhill.py hab_nmpc_mpcsim.py mpcsim_dashboard.py \
            htr_nmpc_mpcsim.py customconstraints.py sstargexample.py \
            softconstraints.py

DOC_TEX := $(addprefix doc/, install.tex cheatsheet.tex introslides.tex \
             octave-vs-python.tex)
DOC_PDF := $(DOC_TEX:.tex=.pdf)

CSTR_MATLAB_FILES := $(addprefix cstr-matlab/, main.m massenbal.m \
                       massenbalstst.m partial.m)

MISC_FILES := COPYING.txt mpctoolssetup.py cstr.m README.md README.pdf

MPC_TOOLS_CASADI_FILES := $(MPCTOOLS_SRC) $(EXAMPLES) $(DOC_PDF) \
                          $(CSTR_MATLAB_FILES) $(MISC_FILES)

ZIPNAME_2 := MPCTools-Python2.zip
ZIPNAME_3 := MPCTools-Python3.zip
ZIPNAMES := $(ZIPNAME_2) $(ZIPNAME_3)
ZIPDIRS := $(basename $(ZIPNAMES))

# Define zip rules.
$(ZIPNAME_3) : $(MPC_TOOLS_CASADI_FILES)
	@echo "Building zip distribution for Python 3."
	@./makezip.py --name $@ $^

$(ZIPNAME_2) : $(MPC_TOOLS_CASADI_FILES)
	@echo "Building zip distribution for Python 2."
	@./makezip.py --python2 --name $@ $^

$(ZIPNAMES) : makezip.py

$(ZIPDIRS) : % : %.zip
	@echo "Unzipping $<"
	@rm -rf $@/*
	@unzip -q $<

.PHONY : $(ZIPDIRS)

UPLOAD_COMMAND := POST https://api.bitbucket.org/2.0/repositories/rawlings-group/mpc-tools-casadi/downloads
define do-bitbucket-upload
echo "Uploading $< to Bitbucket."
echo -n "Enter Bitbucket username: " && read bitbucketuser && curl -v -u $$bitbucketuser -X $(UPLOAD_COMMAND) -F files=@"$<"
endef

upload2 : $(ZIPNAME_2)
	@$(do-bitbucket-upload)
upload3 : $(ZIPNAME_3)
	@$(do-bitbucket-upload)
upload : upload2 upload3
.PHONY : upload upload2 upload3

# Phony rules.
dist2 : $(ZIPNAME_2)
dist3 : $(ZIPNAME_3)
dist : dist2 dist3
.PHONY : dist dist2 dist3

unzip2 : $(basename $(ZIPNAME_2))
unzip3 : $(basename $(ZIPNAME_3))
.PHONY : unzip2 unzip3

# Rules for documentation pdfs.
$(DOC_PDF) : %.pdf : %.tex doc/mpctools.sty
	@echo "Making $@."
	@doc/latex2pdf.py --display errors --dir $(@D) $<

README.pdf : README.md
	@echo "Making $@."
	@pandoc -o $@ $<

# Rule to make Matlab versions of Octave CSTR example.
$(CSTR_MATLAB_FILES) : cstr.m
	@echo "Making Matlab CSTR files."
	@cd doc && ./matlabify.py

# Documentation dependencies.
doc/introslides.pdf : cstr_octave.pdf cstr_python.pdf vdposcillator_lmpc.pdf \
                      vdposcillator_nmpc.pdf cstr_startup.pdf
doc/cheatsheet.pdf : doc/sidebyside.tex
doc/octave-vs-python.pdf : doc/sidebyside-cstr.tex
doc/install.pdf : 

doc : $(DOC_PDF)
.PHONY : doc

# Define rules for intermediate files.
cstr_octave.pdf : cstr.m
	@echo "Making $@."
	@octave $<

cstr_python.pdf : cstr.py
	@echo "Making $@."
	@$(PYTHON) $< --ioff

vdposcillator_lmpc.pdf vdposcillator_nmpc.pdf : vdposcillator.py
	@echo "Making vdposcillator pdfs."
	@$(PYTHON) $< --ioff

cstr_startup.pdf : cstr_startup.py
	@echo "Making $@."
	@$(PYTHON) $< --ioff

doc/sidebyside.tex : comparison_casadi.py comparison_mtc.py
	@echo "Making $@."
	@cd doc && ./doSourceComparison.py $(@F)

doc/sidebyside-cstr.tex : cstr.m cstr.py
	@echo "Making $@."
	@cd doc && ./doSourceComparison.py $(@F)

# Define cleanup rules.
TEXSUFFIXES := .log .aux .toc .vrb .synctex.gz .snm .nav .out
TEX_MISC := $(foreach doc, $(basename $(DOC_TEX)), $(addprefix $(doc), $(TEXSUFFIXES)))
OTHER_MISC := doc/sidebyside.tex doc/sidebyside-cstr.tex README.pdf
clean :
	@rm -f $(ZIPNAMES) $(TEX_MISC) $(OTHER_MISC)
	@rm -rf $(ZIPDIRS)
.PHONY : clean

PDF_MISC := $(DOC_PDF) cstr_octave.pdf cstr_python.pdf vdposcillator_lmpc.pdf \
            vdposcillator_nmpc.pdf cstr_startup.pdf
realclean : clean
	@rm -f $(PDF_MISC) $(CSTR_MATLAB_FILES)
.PHONY : realclean

# Rules for running tests.
test2 : $(ZIPNAME_2)
	@echo "Running Python 2 tests." 
	@python2 testzip.py $< 1> /dev/null
test3 : $(ZIPNAME_3)
	@echo "Running Python 3 tests." 
	@python3 testzip.py $< 1> /dev/null
test : test2 test3
.PHONY : test test2 test3

# Rules for printing variables.
list-examples :
	@echo $(EXAMPLES)
.PHONY : list-examples

