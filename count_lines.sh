#!/bin/bash
# Count core rlmkit lines (excluding examples/, tests/, docs/)
cd "$(dirname "$0")" || exit 1

echo "rlmkit core line count"
echo "================================"
echo ""

for dir in . prompts runtime; do
  count=$(find "rlmkit/$dir" -maxdepth 1 -name "*.py" -exec cat {} + | wc -l)
  if [ "$dir" = "." ]; then
    label="(root)"
  else
    label="$dir/"
  fi
  printf "  %-16s %5s lines\n" "$label" "$count"
done

echo ""
total=$(find rlmkit -name "*.py" | xargs cat | wc -l)
printf "  Core total:     %5s lines\n" "$total"

echo ""
echo "  (excludes: examples/, tests/, docs/)"
