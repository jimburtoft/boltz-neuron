#!/bin/bash
# Same as setup_inf2.sh (works on Inferentia2 and Trainium2). Kept as the generic name.
exec bash "$(dirname "$0")/setup_inf2.sh" "$@"
