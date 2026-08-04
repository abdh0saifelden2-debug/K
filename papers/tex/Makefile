PAPERS = paper0_two_clocks \
         paper1_closure_theory \
         paper2_subglacial_melt_ceiling \
         paper3_scallop_parity_break \
         paper4_subglacial_hydrology_forecasts \
         paper4a_hydraulic_memory_kernel \
         paper4b_sliding_law_field_test

# Papers in the arXiv upload set (the P4 omnibus is superseded by 4a/4b and
# is retained for provenance only — do not upload it alongside its children).
ARXIV_PAPERS = paper0_two_clocks \
               paper1_closure_theory \
               paper2_subglacial_melt_ceiling \
               paper3_scallop_parity_break \
               paper4a_hydraulic_memory_kernel \
               paper4b_sliding_law_field_test

PDFS = $(addsuffix .pdf,$(PAPERS))

all: $(PDFS)

%.pdf: %.tex preamble.sty
	xelatex -interaction=nonstopmode $*.tex >/dev/null
	xelatex -interaction=nonstopmode $*.tex >/dev/null

# NOTE (source of truth): since commit c2e8784 the submission content
# (NR-promotion paragraphs, P1 §5c/5d/5e, abstract trims) lives in the .tex
# files ONLY — the ../*.md drafts are historical and REGENERATING .tex FROM
# THEM DESTROYS THAT CONTENT. The old implicit rule `%.tex: ../%.md` has
# therefore been removed from the default flow. To regenerate deliberately:
#     make regen-from-md P=paper1_closure_theory     # clobbers $(P).tex!
regen-from-md:
	@test -n "$(P)" || { echo "usage: make regen-from-md P=<paper>"; exit 1; }
	@echo "WARNING: overwriting $(P).tex from ../$(P).md (tex-only content will be lost)"
	python3 md2tex.py ../$(P).md $(P).tex

# Package arXiv source tarballs (tex + preamble + only the figures each paper uses).
arxiv: $(addsuffix .pdf,$(ARXIV_PAPERS))
	@mkdir -p arxiv
	@for p in $(ARXIV_PAPERS); do \
	  rm -rf arxiv/$$p; mkdir -p arxiv/$$p/figures; \
	  cp $$p.tex preamble.sty arxiv/$$p/; \
	  for f in $$(grep -o 'includegraphics\[[^]]*\]{[^}]*}' $$p.tex | sed 's/.*{\(.*\)}/\1/'); do \
	    cp $$f arxiv/$$p/figures/ 2>/dev/null || true; \
	  done; \
	  tar -czf arxiv/$$p.tar.gz -C arxiv/$$p .; \
	  echo "arxiv/$$p.tar.gz: $$(du -h arxiv/$$p.tar.gz | cut -f1)"; \
	done

clean:
	rm -f *.aux *.log *.out

.PHONY: all clean regen-from-md arxiv
