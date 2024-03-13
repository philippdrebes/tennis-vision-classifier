from classifier_gpt4v import MultiImage, OpenAIAPIPayload, OpenAIAPIResponse, create_image_index, classify_gpt4v_threaded

### Support Reloading in Jupyter ###
import classifier_gpt4v
from importlib import reload
reload(classifier_gpt4v)
from classifier_gpt4v import MultiImage, OpenAIAPIPayload, OpenAIAPIResponse, create_image_index, classify_gpt4v_threaded


from pathlib import Path
import logging
import traceback
import json
import pandas as pd
from PIL import Image
import traceback
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s' )
dir_src = Path('/mnt/c/Users/8377/switchdrive/SyncVM/w HSLU S3/BS S3 Computer Vision/tennis-vision-classifier/src')
dir_data = dir_src.parent / 'data'
(dir_data / 'debugging_images').mkdir(parents=True, exist_ok=True)

with open("/mnt/c/Users/8377/switchdrive/SyncVM/.env", 'r') as file:   # SECRETS FROM FILE
    env_vars = json.load(file)
    API_KEY_OPENAI   = env_vars["API_KEY_OPENAI"]



def TEST_create_test_MultiImage(read_in_head_limit = 4):
    dir_images  = dir_src.parent / "video/frames/"
    create_image_index(dir_images).to_csv(dir_data / "index_image_filepaths.csv", index=False)

    # image_index_sample = pd.read_csv(dir_data / "index_image_filepaths.csv", index_col=False).head(read_in_head_limit)   # Sample first 25
    image_index_sample = pd.read_csv(dir_data / "index_image_filepaths.csv", index_col=False).sample(read_in_head_limit)   # Sample random 25
    print(image_index_sample)

    input_images_list = list(zip(image_index_sample['image_ID_index'], image_index_sample['image_relative_filepath']))
    multi_img_instance = MultiImage(input_images_list)
    multi_img_instance.image_resize_crop_markup(input_images_list[-1][0], input_images_list[-1][1]).save(dir_data / "debugging_images/test.jpg", "JPEG")   # For debugging just to demonstrate how the method is used
    multi_img_instance.image_combined.save(dir_data / "debugging_images/test_combined.jpg", "JPEG")    # Save a copy of the output image
    from IPython.display import display, Image as IPythonImage
    display(IPythonImage(filename=dir_data / "debugging_images/test_combined.jpg"))
    return multi_img_instance.image_combined


def TEST_create_payload_non_vision():
    # Checks non_vision payloads by default use model='gpt-3.5-turbo', use response_format, and can update params e.g. max_tokens
    try:
        test_payload_non_vision = OpenAIAPIPayload(
            classification_system_message="System message for chat",
            classification_user_text="User message for chat",
            classification_image_list=[],   # Empty list, or could be None
            max_tokens=300
                )
        test_payload_non_vision.max_tokens = 200
        test_payload_non_vision.run()   # Re-init to update max_tokens param

        logging.info(f'test_payload_non_vision:\n{test_payload_non_vision.payload}')

        expected_payload_non_vision = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.0,
            'max_tokens': 200,
            'response_format': {'type': 'json_object'},
            'messages': [
                {'role': 'system', 'content': 'System message for chat'},
                {'role': 'user', 'content': 'User message for chat'}
                ], }
        logging.info(f'expected_payload_non_vision:\n{expected_payload_non_vision}')

        assert test_payload_non_vision.payload == expected_payload_non_vision, "Chat payload does not match expected output."
        logging.info("TEST_payload_non_vision PASSED🎉\n")
    except Exception as e:
        logging.error(f"TEST_payload_non_vision FAILED: {e}\n{traceback.format_exc()}")


def TEST_create_payload_vision():
    # Test case for vision payload with dummy image paths
    try:
        image = Image.open(dir_data / "debugging_images/test_combined.jpg")
        test_payload_vision = OpenAIAPIPayload(
            classification_system_message="System message for vision",
            classification_user_text="User message for vision",
            classification_image_list=[image , image , image ],
            max_tokens=300
                )
        logging.info(f'test_payload_vision:\n{test_payload_vision.payload}')
        image_encoded = OpenAIAPIPayload.encode_image_to_base64(image)

        expected_payload_vision = {
            'model': 'gpt-4-vision-preview',
            'temperature': 0.0,
            'max_tokens': 300,
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'System message for vision'},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_encoded}'}},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_encoded}'}},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_encoded}'}},
                ]}
                        ] }
        logging.info(f'expected_payload_vision:\n{expected_payload_vision}')

        assert test_payload_vision.payload == expected_payload_vision, "Vision payload does not match expected output."
        logging.info("TEST_payload_vision PASSED🎉\n")
    except Exception as e:
        logging.error(f"TEST_payload_vision FAILED: {e}\n{traceback.format_exc()}")


def TEST_run_json_fix_extract_response_test(initial_valid_json, max_tokens=200):
    # Setup the test instance with a dummy payload
    system_message = "Fix any syntax errors and return the same content in valid JSON syntax with no newlines. If any fields are truncated or incomplete, drop them."
    test_text = '{"201": 0, "30": 1, "185": 0, "163": 0, "59": 1, "98": 1, "77": 0, "29": 1, "121'  # Intentionally broken JSON
    payload_test = OpenAIAPIPayload(
        classification_system_message=system_message,
        classification_user_text=test_text,
        classification_image_list=None,  # No images, non-vision request
        max_tokens=max_tokens
        )
    test_instance = OpenAIAPIResponse(payload=payload_test)
    test_instance.response_was_valid_json = initial_valid_json
    test_instance.time_taken = 0
    test_instance.cost_tokens_input = 100
    test_instance.cost_tokens_output = 100

    test_instance.json_fix_extract_response(test_text)
    logging.info(f"Initial valid JSON: {initial_valid_json}, Max Tokens: {max_tokens}")
    logging.info(f"Test Response Extracted: {test_instance.response_extracted}")
    return test_instance.response_extracted

def TEST_json_fix_extract_response_broken_input():
    # Takes a truncated json and fixes it
    logging.info("\n\nRunning TEST_json_fix_extract_response_broken_input")
    TEST_run_json_fix_extract_response_test(initial_valid_json=True)

def TEST_json_fix_extract_response_recursion_catch():
    # Ensures response_was_valid_json is True   which simulates a recursion scenario so is meant to interrupt
    logging.info("\n\nRunning TEST_json_fix_extract_response_recursion_catch")
    TEST_run_json_fix_extract_response_test(initial_valid_json=False)

def TEST_json_fix_extract_response_too_few_max_tokens():
    # max_tokens=50 to check the max_token lengthening works
    logging.info("\n\nRunning TEST_json_fix_extract_response_too_few_max_tokens")
    TEST_run_json_fix_extract_response_test(initial_valid_json=None, max_tokens=25)



def TEST_payload_non_vision_api_interaction():
    try:
        # System message and user text for non-vision payload
        system_message = "Fix any syntax errors and return the same content in valid JSON syntax with no newlines. If any fields are truncated or incomplete, drop them."
        test_text = '{"201": 0, "30": 1, "185": 0, "163": 0, "59": 1, "98": 1, "77": 0, "29": 1, "121'

        payload_test = OpenAIAPIPayload(
            classification_system_message=system_message,
            classification_user_text=test_text,
            classification_image_list=None,  # No images, non-vision request
            max_tokens=25
            )
        # payload_test.max_tokens = 150   # Increase the token limit in case the problem was truncation

        logging.info(f"Initial payload_test payload:\n{json.dumps(payload_test.payload, indent=2)}")
        response_handler = OpenAIAPIResponse(payload=payload_test.payload)
        response_handler.handle_openai_request()
        logging.info(f'TEST response:\n{json.dumps(response_handler.response, indent=2)}')
        logging.info(f'\nTEST response_extracted:\n{json.dumps(response_handler.response_extracted, indent=2)}')

        # Assert checks for expected structure or values in response

        logging.info("TEST_payload_non_vision_api_interaction    no exceptions, check for errors")
    except Exception as e:
        logging.error(f"TEST_payload_non_vision_api_interaction FAILED: {e}\n{traceback.format_exc()}")



def TEST_payload_vision_api_interaction():
    try:
        image1 = Image.open(dir_data / "debugging_images/test_combined_test1.jpg")
        image2 = Image.open(dir_data / "debugging_images/test_combined_test2.jpg")
        image3 = Image.open(dir_data / "debugging_images/test_combined_test3.jpg")
        test_payload_vision = OpenAIAPIPayload(
            classification_system_message="Classify each sub-frame as PLAYING (with label 1) or NON-PLAYING (with label 0): PLAYING scenes only show the tennis court from the standard broadcast angle, fully visible and vertically & horizontally aligned with the court lines (the audience should at most be barely visible). NON-PLAYING scenes show any other content that is not a standard broadcast view of the court, including low-angle, court-side, zoom-in, or audience shots. Respond just in a JSON using the centered number (white text with black outline) in each square as the index for each frame.",
            classification_user_text="Classify each sub-frame as PLAYING (with label 1) or NON-PLAYING (with label 0): PLAYING scenes only show the tennis court from the standard broadcast angle, fully visible and vertically & horizontally aligned with the court lines (the audience should at most be barely visible). NON-PLAYING scenes show any other content that is not a standard broadcast view of the court, including low-angle, court-side, zoom-in, or audience shots. Respond just in a JSON using the centered number (white text with black outline) in each square as the index for each frame.",
            classification_image_list=[image1 , image2 , image3 ],
            max_tokens=300
                )
        logging.info(f'test_payload_vision:\n{test_payload_vision.payload}')

        response_handler = OpenAIAPIResponse(payload=test_payload_vision.payload)
        response_handler.handle_openai_request()
        logging.info(f"API response extracted:\n{response_handler.response_extracted}")

        flattened_classification_results = {}
        for key, value in response_handler.response_extracted.items():
            if key.isdigit():  # This selects imageIDs assuming they are always the only keys that are integers
                flattened_classification_results[int(key)] = value
        logging.info(f"flattened_classification_results:\n{flattened_classification_results}")
        expected_classification = {269: 0, 239: 0, 120: 1, 95: 1, 308: 0, 149: 1, 284: 0, 102: 1, 171: 1, 241: 0, 132: 1, 189: 0}
        logging.info(f"expected_classification:\n{expected_classification}")
        assert expected_classification == flattened_classification_results, "Resulting classifications do not match expected values."
        logging.info("TEST_payload_vision_api_interaction    no errors or exceptions")

    except Exception as e:
        logging.error(f"TEST_payload_vision_api_interaction FAILED: {e}\n{traceback.format_exc()}")


def RUN_classify_gpt4_threaded(batch_size=9):
    create_image_index(dir_src.parent / "video/frames/").to_csv(dir_data / "index_image_filepaths.csv", index=False)
    read_in_head_limit = None   # set to None to read in all values
    images_index_input  = dir_data / "index_image_filepaths.csv"
    images_index_output = dir_data / f"index_image_filepaths_classifications_gpt4v_{batch_size}x{batch_size}.csv"
    updated_df = classify_gpt4v_threaded(batch_size, read_in_head_limit, images_index_input, images_index_output)
    logging.warning(f"updated_df:\n{updated_df}")

def RUN_combine_classifications():
    df_combined = pd.DataFrame()
    standard_cols = ['image_ID_index', 'train_or_test', 'classifier_class', 'image_relative_filepath']
    for file_path in dir_data.glob('index_image_filepaths_classifications_*.csv'):
        temp_df = pd.read_csv(file_path)
        classification_type = file_path.stem.split('_classifications_')[-1]

        if 'gpt4v' in classification_type:
            number = classification_type.split('_')[-1]
            new_col_name = f'classification_gpt4v_{number}'
            temp_df.rename(columns={'classification_gpt4v': new_col_name}, inplace=True)
        elif 'cnn' in classification_type or 'autoencoder' in classification_type:
            # No action needed as we're using the original column names
            pass
        else:
            continue  # Skip files that do not match the expected naming pattern
        if df_combined.empty:
            df_combined = temp_df
        else:
            # We want to merge on the 'image_ID_index' and other standard columns, and keep everything aligned
            df_combined = pd.merge(df_combined, temp_df, on=standard_cols, how='outer')

    # Reorder columns so that 'image_relative_filepath' is the last column
    for col in df_combined.columns:
        if df_combined[col].dtype == float:
            if all(df_combined[col].dropna().apply(float.is_integer)):
                df_combined[col] = df_combined[col].astype(pd.Int64Dtype())  # Use Int64Dtype to handle NaN values properly
    final_cols = [col for col in df_combined.columns if col != 'image_relative_filepath'] + ['image_relative_filepath']
    df_combined = df_combined[final_cols]
    df_combined.to_csv(dir_data / 'index_image_filepaths_classifications_all_combined.csv', index=False)

def RUN_setup_confusion_dash():

    def calculate_confusion_stats(classifications_all_combined):
        df_combined = pd.read_csv(classifications_all_combined)

        classifiers = [col for col in df_combined.columns if col.startswith('classification_') and col != 'classifier_class']
        # classifiers = ['classification_autoencoder', 'classification_cnn', 'classification_gpt4v_16x16', 'classification_gpt4v_25x25', 'classification_gpt4v_9x9']

        # Assert no NaNs
        nan_locations = df_combined[classifiers].isnull()
        if nan_locations.any().any():
            logging.info(f'{df_combined[df_combined[classifiers].isnull().any(axis=1)]}')
            nan_indices = nan_locations[nan_locations.any(axis=1)].stack().index.tolist()
            error_message = f"NaN values found at the following locations: {nan_indices}"
            raise AssertionError(error_message)


        confusion_stats = {}

        for classifier in classifiers:
            y_true = df_combined['classifier_class']
            y_pred = df_combined[classifier].astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total = tn + fp + fn + tp
            tn_pct, fp_pct, fn_pct, tp_pct = (np.round(tn / total,3), np.round(fp / total,3), np.round(fn / total,3), np.round(tp / total,3))
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            matthews = matthews_corrcoef(y_true, y_pred)

            confusion_stats[classifier] = {
                "True Positive": tp, "False Positive": fp, "False Negative": fn, "True Negative": tn,
                "True Positive %": tp_pct, "False Positive %": fp_pct, "False Negative %": fn_pct, "True Negative %": tn_pct,
                "Accuracy": np.round(accuracy,3),
                "Precision": np.round(precision,3),
                "Recall": np.round(recall,3),
                "F1 Score": np.round(f1,3),
                "Matthews Correlation Coefficient": np.round(matthews,3)
            }

        logging.info(f"confusion_stats:\n{confusion_stats}")
        for classifier, stats in confusion_stats.items():
            logging.info(f"Classifier: {classifier}")
            for stat_name, value in stats.items():
                logging.info(f"{stat_name:20} {value}")
            logging.info("")  # Adds an extra line for spacing between classifiers

        def convert_np_int64(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError

        output_file_path = classifications_all_combined.parent / "confusion_stats.json"
        with open(output_file_path, 'w') as outfile:
            json.dump(confusion_stats, outfile, indent=4, default=convert_np_int64)
        logging.info(f"Confusion stats saved to {output_file_path}")


    def translate_to_xy(confusion_pair):
        # Translates a confusion pair (ground_truth, prediction) to x, y coordinates.
        mapping = {
            (1, 1): ((0.05, 0.45), (0.55, 0.95)),  # True Positive
            (1, 0): ((0.05, 0.45), (0.05, 0.45)),  # False Negative
            (0, 1): ((0.55, 0.95), (0.55, 0.95)),  # False Positive
            (0, 0): ((0.55, 0.95), (0.05, 0.45)),  # True Negative
            }
        x_bounds, y_bounds = mapping[confusion_pair]
        x = np.round(random.uniform(*x_bounds),3)
        y = np.round(random.uniform(*y_bounds),3)
        return (x, y)

    # Test translate_to_xy with example input
    # confusion_pairs = [(1, 1), (1, 0), (0, 1), (0, 0)]
    # xy_coordinates = [translate_to_xy(pair) for pair in confusion_pairs]
    # logging.info(xy_coordinates)


    def add_coordinates(classifications_all_combined):
        df = pd.read_csv(classifications_all_combined)
        logging.info(f'df.columns: {df.columns}')
        for classifier in [col for col in df.columns if col.startswith('classification_')]:
            logging.info(f"Adding coordinates for {classifier}")
            coord_x_col = f"{classifier}_coord_x"
            coord_y_col = f"{classifier}_coord_y"

            df[coord_x_col] = np.nan
            df[coord_y_col] = np.nan

            for index, row in df.iterrows():
                ground_truth = row['classifier_class']
                prediction = row[classifier]
                confusion_pair = (ground_truth, prediction)
                x, y = translate_to_xy(confusion_pair)
                df.at[index, coord_x_col] = x
                df.at[index, coord_y_col] = y

        return df


    classifications_all_combined = dir_data / 'index_image_filepaths_classifications_all_combined.csv'
    calculate_confusion_stats(classifications_all_combined)
    df_coords = add_coordinates(classifications_all_combined)
    df_coords.to_csv(dir_data / 'index_image_filepaths_classifications_all_combined_coords.csv', index=False)

# test_multi_img = TEST_create_test_MultiImage(read_in_head_limit = 4)
# TEST_create_payload_non_vision()
# TEST_create_payload_vision()
# TEST_json_fix_extract_response_broken_input()
# TEST_json_fix_extract_response_recursion_catch()
# TEST_json_fix_extract_response_too_few_max_tokens()
# TEST_payload_non_vision_api_interaction()
# TEST_payload_vision_api_interaction()
# RUN_classify_gpt4_threaded(9)
RUN_combine_classifications()
RUN_setup_confusion_dash()
