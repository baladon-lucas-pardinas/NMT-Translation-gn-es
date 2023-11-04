These scripts were used to fix errors in the results:

1. **remove_noisy_data.ipynb**: This script is used to remove metrics calculated
from bugged models or runs where the server failed.
2. **replace_concatenated_lines**: This script is used to fix an error in the
ingestion phase where we accidentally concatenated 10 lines. It is also
added a final line which was missing from the decoded output.