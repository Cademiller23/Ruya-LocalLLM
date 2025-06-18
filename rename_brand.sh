#!/bin/bash

# Usage: bash rename_brand.sh
# Description: Replaces all instances of Ruya-LLM branding with RuyaLLM.

find . -type f \
  -not -path '*/node_modules/*' \
  -not -path '*/.git/*' \
  -not -path '*.png' \
  -not -path '*.jpg' \
  -not -path '*.jpeg' \
  -not -path '*.ico' \
  -not -path '*.svg' \
  -exec grep -Il '.' {} \; | while read -r file; do
    echo "Updating: $file"
    sed -i '' -e 's/ruya-llm/ruya-llm/g' "$file"
    sed -i '' -e 's/Ruya-LLM/Ruya-LLM/g' "$file"
    sed -i '' -e 's/RuyaLLM/RuyaLLM/g' "$file"
done