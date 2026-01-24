import os, sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_annotations(annotations_file_path: str = "data/metadata.pkl") -> dict:
    """
    Extract annotations from a pickle file.

    Parameters
    ----------
    annotations_file_path (str): 
        Path to the pickle file containing annotations.
    
    Returns
    -------
    dict:
        Dictionary containing the extracted annotations.
    """    
    with open(annotations_file_path, 'rb') as f:
        metadata = pickle.load(f)
        
    return metadata

def ensure_annotations_complete(annotations: dict, data_dir: str = "data/images") -> bool:
    file_names = annotations['file_name']
    split_origins = annotations['split_origin']
    words_list = annotations['words']
    bboxes_list = annotations['bboxes']
    full_texts = annotations['full_text']
    
    for i, file_name in enumerate(file_names):
        # Check that the file exists in the data directory
        if not os.path.isfile(os.path.join(data_dir, file_name)):
            raise FileNotFoundError(f"File {file_name} not found in {data_dir}")
        # Check that the split_origin, words, bboxes, full_text entries are not empty
        if not split_origins[i]:
            raise ValueError(f"Split origin missing for file {file_name}")
        if not words_list[i]:
            raise ValueError(f"Words list missing for file {file_name}")
        if not bboxes_list[i]:
            raise ValueError(f"BBoxes list missing for file {file_name}")
        if not full_texts[i]:
            raise ValueError(f"Full text missing for file {file_name}")
    return True




if __name__ == "__main__":
    annotations = extract_annotations()
    try:
        if ensure_annotations_complete(annotations):
            print("All annotations are complete and valid.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Annotation validation error: {e}")
        
        