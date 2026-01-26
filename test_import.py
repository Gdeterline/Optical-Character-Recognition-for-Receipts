import sys
import os
sys.path.append(os.path.abspath('src'))
try:
    from preprocessing.utils import crop_words_from_receipts
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    # inspect what is in preprocessing.utils
    import preprocessing.utils
    print("Dir of preprocessing.utils:", dir(preprocessing.utils))
