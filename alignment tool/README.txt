Tool1："MAGPIE registration quick guide"
Goal: click landmarks → estimate affine/TPS transform with MAGPIE → export aligned MSI coordinates → map to Visium image pixel space (hires/lowres/fullres).
Here is some references:
  1."paper":https://www.nature.com/articles/s41467-025-68003-w.pdf
  2."MAGPIE Pipeline":https://core-bioinformatics.github.io/magpie/intro.html
  3."Shiny landmarks":https://core-bioinformatics.github.io/magpie/shiny-app/shiny-app.html
  4."Snakemake Rule 1":https://core-bioinformatics.github.io/magpie/snakemake-rules/rule-1.html

Tool2："SLAT adjacent-slice alignment quick guide"
Goal: align two adjacent slices (or two modalities/slices) with SLAT → get matched pairs → build an “RNA-on-ATAC coordinate” AnnData → (optionally pad missing spots/cells with zeros).
Here is some references:
  1."paper":https://www.nature.com/articles/s41467-023-43105-5.pdf
  2."SLAT Pipeline":https://github.com/gao-lab/SLAT
  3."tutorials":https://slat.readthedocs.io/en/latest/tutorials.html (You can use this guide to align adjacent slices)