#!/bin/bash

# Check if a ticker argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <TICKER> [OPTIONS]"
  echo "Example: $0 CBA --wacc 0.08 --margin 0.12"
  exit 1
fi

# The first argument is the ticker
TICKER="$1"

# Shift the first argument away so that all other arguments ($2, $3, ...) 
# are passed directly to the ainalyst command
shift

# Run the ainalyst CLI command with the ticker and all other passed options
# The output will include logs to stdout and an HTML report will be generated.
# The --output is set to ensure reports go to a standard location.
# If you want to customize the output path, you can pass --output via OPTIONS.
ainalyst analyse "$TICKER" "$@" --output "reports/${TICKER}.html"

echo "\nAnalysis complete. Report saved to reports/${TICKER}.html"
