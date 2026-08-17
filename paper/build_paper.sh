#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
paper_dir="$repo_dir/paper"
build_dir="$repo_dir/tmp/texbuild/main"
output_dir="$paper_dir"

mkdir -p "$build_dir" "$output_dir"

export SOURCE_DATE_EPOCH=1786942800
export FORCE_SOURCE_DATE=1
export TZ=UTC
export TEXINPUTS="$paper_dir//:"
export BIBINPUTS="$paper_dir:"

pdflatex_cmd=${PDFLATEX:-pdflatex}
bibtex_cmd=${BIBTEX:-bibtex}

cd "$paper_dir"
"$pdflatex_cmd" -interaction=nonstopmode -halt-on-error -file-line-error \
  -output-directory="$build_dir" main.tex
cd "$build_dir"
"$bibtex_cmd" main
cd "$paper_dir"
"$pdflatex_cmd" -interaction=nonstopmode -halt-on-error -file-line-error \
  -output-directory="$build_dir" main.tex
"$pdflatex_cmd" -interaction=nonstopmode -halt-on-error -file-line-error \
  -output-directory="$build_dir" main.tex

cp "$build_dir/main.pdf" "$output_dir/main.pdf"
echo "$output_dir/main.pdf"
