#!/bin/bash
# compile.sh — compile paper_clean.tex with paper_clean.bbl (no bibtex needed)
# Usage: ./compile.sh  ou  bash compile.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

TEX="paper_clean.tex"
PDF="paper_clean.pdf"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Erreur: pdflatex introuvable (texlive-base manquant)" >&2
  exit 1
fi

if [ ! -f "$TEX" ]; then
  echo "Erreur: $TEX introuvable dans $DIR" >&2
  exit 1
fi

if [ ! -f "paper_clean.bbl" ]; then
  echo "Attention: paper_clean.bbl manquant — citations non résolues" >&2
fi

# stubs locaux si texlive-latex-extra/science non installés (évite un apt de 500 Mo)
if ! kpsewhich multirow.sty >/dev/null 2>&1 && [ ! -f "multirow.sty" ]; then
  echo "Note: multirow.sty absent système → stub local créé"
  cat > multirow.sty <<'EOS'
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{multirow}[2024/01/01 stub]
\newcommand{\multirow}[3][c]{\begin{tabular}[#1]{@{}c@{}}#3\end{tabular}}
\newcommand{\multirowsetup}{\relax}
EOS
fi

if ! kpsewhich siunitx.sty >/dev/null 2>&1 && [ ! -f "siunitx.sty" ]; then
  echo "Note: siunitx.sty absent système → stub local créé (non utilisé)"
  cat > siunitx.sty <<'EOS'
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{siunitx}[2024/01/01 stub]
\DeclareOption*{\relax}\ProcessOptions\relax
\newcommand{\SI}[2]{#1\,#2}
\newcommand{\si}[1]{#1}
\newcommand{\num}[1]{#1}
\newcommand{\qty}[2]{#1\,#2}
\newcommand{\sisetup}[1]{}
EOS
fi

# Erreur stricte si une figure requise est manquante — jamais de placeholder
missing=0
for f in results/figures/layerwise_crowd.pdf results/figures/layerwise_emotwics.pdf results/figures/conditional_gain.pdf results/figures/multiprot_sites.pdf; do
  if [ ! -f "$f" ]; then
    echo "Erreur: figure manquante $f — compilation annulée (aucun placeholder ne sera généré)" >&2
    missing=1
  fi
done
# Vérification étendue : toute figure \includegraphics dans le .tex doit exister
while IFS= read -r fig; do
  [ -z "$fig" ] && continue
  if [ ! -f "$fig" ] && [ ! -f "${fig}.pdf" ]; then
    echo "Erreur: figure manquante $fig (référencée dans $TEX) — compilation annulée" >&2
    missing=1
  fi
done < <(grep -F 'includegraphics' "$TEX" 2>/dev/null | sed -E 's/.*\{([^}]+)\}.*/\1/' || true)
if [ "$missing" -ne 0 ]; then
  echo "Erreur: une ou plusieurs figures sont manquantes — $PDF ne sera pas généré" >&2
  exit 1
fi

echo "→ pdflatex passe 1/3..."
pdflatex -interaction=nonstopmode "$TEX" >/dev/null

echo "→ pdflatex passe 2/3..."
pdflatex -interaction=nonstopmode "$TEX" >/dev/null

echo "→ pdflatex passe 3/3..."
pdflatex -interaction=nonstopmode "$TEX" >/dev/null

if [ -f "$PDF" ]; then
  echo "✓ $PDF généré ($(du -h "$PDF" | cut -f1), $(pdfinfo "$PDF" 2>/dev/null | awk '/Pages:/{print $2" pages"}'))"
else
  echo "Erreur: $PDF non généré" >&2
  exit 1
fi
