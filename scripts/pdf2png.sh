#!/bin/zsh

# Converts all PDFs in the current directory to PNGs
# Output goes into ./pdf_png/

input_dir="$PWD"
output_dir="$input_dir/pdf_png"

mkdir -p "$output_dir"

echo "Input folder:  $input_dir"
echo "Output folder: $output_dir"
echo ""

# Loop through all PDFs (case-insensitive)
for pdf in "$input_dir"/*.pdf(N) "$input_dir"/*.PDF(N); do
  # Skip if no files
  [ -e "$pdf" ] || continue

  base="$(basename "$pdf")"
  name="${base%.*}"

  echo "Converting: $base"
  sips -s format png "$pdf" --out "$output_dir/$name.png"
done

echo ""
echo "✅ Done! PNG files are in: $output_dir"
