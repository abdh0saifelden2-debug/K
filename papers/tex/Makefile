PAPERS = paper1_closure_theory \
         paper2_subglacial_melt_ceiling \
         paper3_scallop_parity_break \
         paper4_subglacial_hydrology_forecasts \
         paper4a_hydraulic_memory_kernel \
         paper4b_sliding_law_field_test

PDFS = $(addsuffix .pdf,$(PAPERS))

all: $(PDFS)

%.pdf: %.tex preamble.sty
	xelatex -interaction=nonstopmode $*.tex >/dev/null
	xelatex -interaction=nonstopmode $*.tex >/dev/null

%.tex: ../%.md md2tex.py
	python3 md2tex.py $< $@

clean:
	rm -f *.aux *.log *.out

.PHONY: all clean
