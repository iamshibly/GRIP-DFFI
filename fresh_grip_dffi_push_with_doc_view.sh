#!/bin/bash
set -e

SOURCE_DIR="$(pwd)"
CLEAN_DIR="$HOME/Desktop/GRIP-DFFI_CLEAN_PUSH"
REPO_URL="https://github.com/iamshibly/GRIP-DFFI.git"
MAX_BYTES=26214400

echo "Source folder:"
echo "$SOURCE_DIR"
echo ""
echo "Clean push folder:"
echo "$CLEAN_DIR"
echo ""

rm -rf "$CLEAN_DIR"
mkdir -p "$CLEAN_DIR"

echo "Copying your local files..."
rsync -a \
  --exclude ".git" \
  --exclude ".DS_Store" \
  --exclude "docs" \
  --exclude "*.html" \
  --exclude "FILES_OVER_25MB_NOT_PUSHED.txt" \
  --exclude "GRIP-DFFI_github_ready" \
  --exclude "GRIP-DFFI_CLEAN_PUSH" \
  "$SOURCE_DIR"/ "$CLEAN_DIR"/

cd "$CLEAN_DIR"

echo "Cleaning old generated junk..."
rm -rf docs
rm -f FILES_OVER_25MB_NOT_PUSHED.txt
find "$CLEAN_DIR" -name "*.html" -type f -delete

echo "Removing files over 25 MB..."
find "$CLEAN_DIR" -type f -print0 | while IFS= read -r -d '' file; do
  size=$(stat -f%z "$file")
  if [ "$size" -gt "$MAX_BYTES" ]; then
    echo "Removed over 25 MB: ${file#$CLEAN_DIR/}"
    rm -f "$file"
  fi
done

echo "Installing needed Python converters..."
python3 -m pip install --user nbconvert jupyter pymupdf >/dev/null 2>&1 || true

echo "Creating GitHub-readable Markdown output files for notebooks..."
find "$CLEAN_DIR" \
  -name "*.ipynb" \
  ! -path "*/.ipynb_checkpoints/*" \
  -print0 | while IFS= read -r -d '' nb; do

  dir="$(dirname "$nb")"
  base="$(basename "$nb" .ipynb)"

  echo "Converting notebook: ${nb#$CLEAN_DIR/}"

  python3 -m jupyter nbconvert \
    --to markdown \
    "$nb" \
    --output "${base}_OUTPUT_VIEW.md" \
    --output-dir "$dir" || true
done

echo "Finding supplementary DOCX..."
SUPP_DOCX_FOUND="$(find "$CLEAN_DIR" -iname "*Supplementary*.docx" -print | head -n 1)"

if [ -z "$SUPP_DOCX_FOUND" ]; then
  SUPP_DOCX_FOUND="$(find "$CLEAN_DIR" -iname "*.docx" -print | head -n 1)"
fi

if [ -z "$SUPP_DOCX_FOUND" ]; then
  echo "No DOCX found. Skipping supplementary document conversion."
else
  echo "Found DOCX:"
  echo "$SUPP_DOCX_FOUND"

  SUPP_BASE="Supplementary_Document_S1_Complete_GRIP_DFFI_Validation_Tables_and_Robustness_Analyses"
  SUPP_DOCX="$CLEAN_DIR/${SUPP_BASE}.docx"
  SUPP_PDF="$CLEAN_DIR/${SUPP_BASE}.pdf"
  SUPP_MD="$CLEAN_DIR/Supplementary_Document_S1_VIEW.md"
  SUPP_PAGES_DIR="$CLEAN_DIR/Supplementary_Document_S1_pages"

  if [ "$SUPP_DOCX_FOUND" != "$SUPP_DOCX" ]; then
    cp "$SUPP_DOCX_FOUND" "$SUPP_DOCX"
  fi

  echo "Converting supplementary DOCX to PDF..."

  TMP_PDF_DIR="$(mktemp -d)"

  if command -v libreoffice >/dev/null 2>&1; then
    libreoffice --headless --convert-to pdf --outdir "$TMP_PDF_DIR" "$SUPP_DOCX" || true
  elif [ -x "/Applications/LibreOffice.app/Contents/MacOS/soffice" ]; then
    /Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to pdf --outdir "$TMP_PDF_DIR" "$SUPP_DOCX" || true
  else
    echo ""
    echo "LibreOffice is not installed, so PDF/page-view cannot be created automatically."
    echo "Install LibreOffice from https://www.libreoffice.org/download/download-libreoffice/ and rerun this script."
    echo ""
  fi

  GENERATED_PDF="$(find "$TMP_PDF_DIR" -iname "*.pdf" -print | head -n 1)"

  if [ -n "$GENERATED_PDF" ]; then
    cp "$GENERATED_PDF" "$SUPP_PDF"

    echo "Rendering supplementary PDF pages to images..."
    rm -rf "$SUPP_PAGES_DIR"
    mkdir -p "$SUPP_PAGES_DIR"

    export SUPP_PDF
    export SUPP_PAGES_DIR

    python3 - <<'PY'
import os
from pathlib import Path
import fitz

pdf_path = Path(os.environ["SUPP_PDF"])
pages_dir = Path(os.environ["SUPP_PAGES_DIR"])
pages_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(str(pdf_path))

for i, page in enumerate(doc, start=1):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    out = pages_dir / f"page_{i:03d}.png"
    pix.save(str(out))

print(f"Rendered {len(doc)} pages.")
PY

    echo "Creating GitHub-viewable supplementary Markdown file..."

    cat > "$SUPP_MD" <<MD
# Supplementary Document S1

## Complete GRIP-DFFI Validation Tables and Robustness Analyses

GitHub may not preview the Word document correctly.  
Use this Markdown view to read the supplementary document directly in GitHub.

Download files:

- [Download DOCX](${SUPP_BASE}.docx)
- [Download PDF](${SUPP_BASE}.pdf)

MD

    page_num=1
    find "$SUPP_PAGES_DIR" -name "page_*.png" | sort | while read -r img; do
      img_name="$(basename "$img")"
      echo "" >> "$SUPP_MD"
      echo "## Page $page_num" >> "$SUPP_MD"
      echo "" >> "$SUPP_MD"
      echo "![](Supplementary_Document_S1_pages/$img_name)" >> "$SUPP_MD"
      page_num=$((page_num + 1))
    done
  else
    echo "PDF was not created. Supplementary Markdown page view skipped."
  fi

  rm -rf "$TMP_PDF_DIR"
fi

echo "Removing any generated files over 25 MB..."
find "$CLEAN_DIR" -type f -print0 | while IFS= read -r -d '' file; do
  size=$(stat -f%z "$file")
  if [ "$size" -gt "$MAX_BYTES" ]; then
    echo "Removed over 25 MB: ${file#$CLEAN_DIR/}"
    rm -f "$file"
  fi
done

echo "Creating clean short README.md..."
cat > "$CLEAN_DIR/README.md" <<'README'
# GRIP-DFFI

Important note: GitHub may not directly render some large `.ipynb` notebook files or `.docx` documents correctly.

## How to View Notebook Outputs

For each experiment folder, open the file ending with:

`_OUTPUT_VIEW.md`

Those files show the notebook code and outputs directly in GitHub.

The `.ipynb` files are kept as the original source notebooks.

## Supplementary Document

To view the supplementary document directly in GitHub, open:

[Supplementary_Document_S1_VIEW.md](Supplementary_Document_S1_VIEW.md)

The original Word/PDF versions are also included for download.
README

echo "Removing empty folders..."
find "$CLEAN_DIR" -type d -empty -delete

echo "Initializing fresh Git repository..."
rm -rf "$CLEAN_DIR/.git"
git init
git branch -M main
git remote add origin "$REPO_URL"

echo "Adding files..."
git add -A

echo "Committing..."
git commit -m "Fresh clean upload with GitHub-viewable notebook and supplementary document outputs"

echo "Force pushing clean repository..."
git push --force origin main

echo ""
echo "DONE."
echo "Now open:"
echo "https://github.com/iamshibly/GRIP-DFFI"
echo ""
echo "For the supplementary document, open:"
echo "Supplementary_Document_S1_VIEW.md"
