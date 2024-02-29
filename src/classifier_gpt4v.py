

# gpt4v  list of image uri's goes in, classifications go out, classifications are cached




# from import

# os.chdir('/mnt/c/Users/8377/switchdrive/SyncVM/w HSLU S3/BS S3 Computer Vision/playing_waiting_classifier_images')
# playing_dir_0 = './data_PCA/playing_waiting_frames_sorted_playing=0'
# playing_dir_1 = './data_PCA/playing_waiting_frames_sorted_playing=1'
# playing_dir_1_test = f'{playing_dir_1}_test'

# width = 80    # performs better than 320 or 160
# height = 45   # performs better than 180 or  90

# pca_model_path = f'./model_pca/pca_model_{width}x{height}.joblib'
# playing_images = load_and_process_images(playing_dir_1, width, height, f'{playing_dir_1}_test')
# pca_and_visualize(playing_images, width, height, pca_model_path)

### new_image_path = playing_dir_0 + '/Emma Raducanu vs Leylah Fernandez Full Match ｜ 2021 US Open Final [F99Kz2eptqM]_00-03-0.jpg'   # playing=0
### new_image_path = playing_dir_1 + '/Emma Raducanu vs Leylah Fernandez Full Match ｜ 2021 US Open Final [F99Kz2eptqM]_00-06-0.jpg'   # playing=1
##### plot_images_and_calculate_similarity_individual(new_image_path, pca_model_path, width, height)
##
##




# import re
# import base64
# import json
# import requests
# import functools
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import functools


# from concurrent.futures import ThreadPoolExecutor
# import functools, re
# def threaded_openai_call(max_workers=10):
#     def decorator_openai_call(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 future = executor.submit(func, *args, **kwargs)
#                 return future.result()
#         return wrapper
#     return decorator_openai_call



# import re
# import base64
# import json
# import requests
# import functools
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import functools


# @threaded_openai_call(max_workers=10)
# @profiler_mem_time_detailed
# def call_openai_api_vision(row, image_512_path, base64_image):
#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY_OPENAI}"}
#     payload = {
#         "model": "gpt-4-vision-preview",
#         "messages": [
#             {"role": "user", "content": [
#         # "text": f"Pytesseract: {row['slide_transcript_pytesseract']}" },
#             {"type": "text", "text": f"Here is a first-shot attempt at transcribing the text, build on this: {row['slide_transcript_easyocr']}. Now transcribe the contents of this image in a structured .json format keeping seperate paragraphs distinct. If there are any non-text visual elements (icons, graphs) describe them. Be terse give no comments on font or colors, only transcribe text & visual elements. Keep all comments inside the .json . Begin and end with curly braces just like a raw .json  "},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
#         ]} ], "max_tokens": 400
#         }
#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#     if response.status_code == 200:
#         response_json = response.json()
#         content = response_json['choices'][0]['message']['content']
#         json_content = re.search(r'{.*}', content, re.DOTALL)
#         json_str = json_content.group(0) if json_content else '{}'
#         token_usage = response_json['usage']
#         slide_number_this_slide = image_512_path.split('/')[1].split('_')[1]
#         return {'filepath': image_512_path, 'pitchdeck_title': image_512_path.split('/')[1].split('_')[0], 'slide_number_this_slide': slide_number_this_slide, 'prompt_tokens': token_usage['prompt_tokens'], 'completion_tokens': token_usage['completion_tokens'], 'total_tokens': token_usage['total_tokens'], 'slide_transcript_easyocr': row['slide_transcript_easyocr'], 'gpt4v_transcription': json_str}
#     else:
#         return {"error": response.text, "image": image_512_path}


# @profiler_mem_time_detailed
# def process_images_and_call_api(df):
#     def encode_image_to_base64(image_path):
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')

#     responses = []
#     df_original = df.loc[df['slide_subimage_type'] == 'original']
#     df_original = df_original   # .head(6)

#     with ThreadPoolExecutor(max_workers=10) as executor:   # try out multiple requests
#         future_to_response = {executor.submit(call_openai_api_vision, row, row['relative_filepath'].replace('original', '512'), encode_image_to_base64(f"./C_data_processed/{row['relative_filepath'].replace('original', '512')}")): row for index, row in df_original.iterrows()}

#     for future in as_completed(future_to_response):
#         row = future_to_response[future]
#         try:
#             response = future.result()
#         except Exception as exc:
#             print(f"{row['relative_filepath']} generated an exception: {exc}")
#         else:
#             responses.append(response)
#     return responses


# def validate_and_fix_json(df):
#     def is_valid_json(json_str):
#         try:
#             json.loads(json_str)
#             return True, None
#         except json.JSONDecodeError as e:
#             return False, str(e)

#     @profiler_mem_time_detailed
#     def fix_json_with_gpt(content, error_message):
#         headers = {"Authorization": f"Bearer {API_KEY_OPENAI}"}
#         payload = {
#             "model": "gpt-3.5-turbo-1106",
#             "response_format": { "type": "json_object" },
#             "messages": [
#                 {"role": "system", "content": "Fix the errors and return the same content in valid JSON syntax."},
#                 {"role": "user", "content": f"Error: {error_message}\nContent: {content}"}
#             ]
#         }
#         response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#         if response.status_code == 200:
#             response_json = response.json()
#             return response_json['choices'][0]['message']['content']
#         else:
#             return "GPT failed to fix JSON: " + response.text

#     if 'gpt4v_transcription_validated' not in df.columns:
#         df['gpt4v_transcription_validated'] = df['gpt4v_transcription']

#     # Convert non-string values to strings
#     df['gpt4v_transcription_validated'] = df['gpt4v_transcription_validated'].apply(lambda x: str(x) if not isinstance(x, str) else x)

#     for index, row in df.iterrows():
#         is_valid, error_msg = is_valid_json(row['gpt4v_transcription_validated'])
#         if not is_valid:
#             fixed_json = fix_json_with_gpt(row['gpt4v_transcription_validated'], error_msg)
#             df.at[index, 'gpt4v_transcription_validated'] = fixed_json








# https://superstudy.guide/algorithms-data-structures/data-structures/stacks-queues/#queues
# Immutable checkpoints are saved after each stage, all outputs return their contents so that the functions can be piped.

# https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import base64
import json
import math
import pandas as pd
import re
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from sklearn.metrics import confusion_matrix
import itertools



dir_src = Path('/mnt/c/Users/8377/switchdrive/SyncVM/w HSLU S3/BS S3 Computer Vision/tennis-vision-classifier/src')
dir_data = dir_src.parent / 'data'
(dir_data / 'debugging_images').mkdir(parents=True, exist_ok=True)

with open("/mnt/c/Users/8377/switchdrive/SyncVM/.env", 'r') as file:   # SECRETS FROM FILE
    env_vars = json.load(file)
    API_KEY_OPENAI   = env_vars["API_KEY_OPENAI"]




### Stage 1: Image Index Preparation ###
### outputs index_image_filepaths.csv = index containing paths to images with some associated metadata

def create_image_index(dir_images):
    # Function to create image index from input folders, assumes dir_images contains subfolders called  test/play   and /test/waiting    and  /train/play  and  /train/waiting
    data_rows = []
    image_ID_index = 0
    subdirs = ['test/play', 'test/waiting']    # subdirs = ['test/play', 'test/waiting', 'train/play', 'train/waiting']

    for subdir in subdirs:
        train_or_test = 'train' if 'train' in subdir else 'test'
        classifier_class = 1 if 'play' in subdir else 0

        for img_path in (dir_images / subdir).glob("*.jpg"):
            image_ID_index += 1
            # parts = img_path.stem.split("_")
            # pitchdeck_title, slide_number, slide_subimage_type = parts[0], parts[1], parts[-1]

            data_rows.append({
                # "pitchdeck_title": pitchdeck_title,
                # "slide_number": slide_number,
                # "slide_subimage_type": slide_subimage_type,
                "image_ID_index": image_ID_index,
                "train_or_test": train_or_test,
                "classifier_class": classifier_class,
                "image_relative_filepath": "../video/" + str(img_path.relative_to(dir_images.parent)),
            })

    return pd.DataFrame(data_rows)





@dataclass
class MultiImage:
    # Takes in references to images, with their IDs, and composits them into a single labelled image
    input_images_list: List[Tuple[int, str]]  # List of tuples (image_ID_index, image_relative_filepath)
    grid_num_rows_cols: Tuple[int] = field(init=True, default=(1))
    image_shurnk_size: Tuple[int] = field(init=True, default=(512))
    image_combined: Image.Image = field(init=False)

    def __post_init__(self):
        self.grid_num_rows_cols = math.ceil(math.sqrt(len(self.input_images_list)))
        self.image_shurnk_size = math.floor(512 / self.grid_num_rows_cols)
        marked_images = [self.image_resize_crop_markup(index, image_filepath) for index, image_filepath in self.input_images_list]
        self.image_combined = self.image_combiner(marked_images)


    def image_resize_crop_markup(self, index, image_filepath):
        font_size = 25
        img = Image.open(image_filepath)

        # Resizes with constant aspect ratio to height 512
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * 512)
        img = img.resize((new_width, 512), Image.LANCZOS)

        # Center crop to 512x512
        if new_width > 512:
            left = (new_width - 512) / 2
            upper = 0
            right = (new_width + 512) / 2
            lower = 512
            img = img.crop((left, upper, right, lower))

        # Resizes again to fit into a minimal square bounding box
        new_size = self.image_shurnk_size
        img = img.resize((new_size, new_size), Image.LANCZOS)

        # Draws a single pixel wide white line across the top and left borders
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (new_size - 1, 0)], fill="white", width=1)  # Top border
        draw.line([(0, 0), (0, new_size - 1)], fill="white", width=1)  # Left border

        font = ImageFont.truetype("../scripts/ARIAL.ttf", font_size)

        text_bbox = draw.textbbox((0, 0), str(index), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        img_width, img_height = img.size
        position = ((img_width - text_width) / 2, (img_height - text_height) / 2)

        # Draw border slightly offset from the original position
        offset_range = [-2, 0, 2]
        for x_offset in offset_range:
            for y_offset in offset_range:
                border_position = (position[0] + x_offset, position[1] + y_offset)
                draw.text(border_position, str(index), font=font, fill="black")

        # Draws the main text in text_colour
        draw.text(position, str(index), font=font, fill="white")
        return img

    def image_combiner(self, marked_images):
        grid_size = self.grid_num_rows_cols
        new_size = self.image_shurnk_size
        canvas = Image.new('RGB', (512, 512), "black")
        for i, img in enumerate(marked_images):
            x = (i % grid_size) * new_size
            y = (i // grid_size) * new_size
            canvas.paste(img, (x, y))
        return canvas




def classify_gpt4v(image):
    # Input: a composite PIL image of frames with index numbers stamped
    # Returns: a JSON of classifications of each frame e.g. {"201": 0, "30": 1, "185": 0, "163": 0, "59": 1, "98": 1, "77": 0, "29": 1, "121": 0, "time_taken": 7.13, "initial_response_was_valid_json": 1, "total_input_tokens": 212, "total_output_tokens": 69}

    def encode_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def call_openai_api_vision(base64_image):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY_OPENAI}"}
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify each sub-frame as PLAYING (with label 1) or NON-PLAYING (with label 0): PLAYING scenes only show the tennis court from the standard broadcast angle, fully visible and vertically & horizontally aligned with the court lines (the audience should at most be barely visible). NON-PLAYING scenes show any other content that is not a standard broadcast view of the court, including low-angle, court-side, zoom-in, or audience shots. Respond just in a JSON using the centered number (white text with black outline) in each square as the index for each frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"} }
                ]}
            ],
            "temperature": 0,
            "max_tokens": 400
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        return response

    def is_valid_json(json_str):
        try:
            json.loads(json_str)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)

    def fix_json_with_gpt(content, error_message):
        headers = {"Authorization": f"Bearer {API_KEY_OPENAI}"}
        payload = {
            "model": "gpt-3.5-turbo-1106",
            "response_format": "json",
            "messages": [
                {"role": "system", "content": "Fix the errors and return the same content in valid JSON syntax."},
                {"role": "user", "content": f"Error: {error_message}\nContent: {content}"}
            ]
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "GPT failed to fix JSON: " + response.text


    start_time = time.time()
    base64_image = encode_image_to_base64(image)
    response = call_openai_api_vision(base64_image)

    initial_response_was_valid_json = 1
    total_input_tokens = total_output_tokens = 0

    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        json_str = re.search(r'{.*}', content, re.DOTALL).group(0) if re.search(r'{.*}', content, re.DOTALL) else '{}'

        is_valid, error_msg = is_valid_json(json_str)
        if not is_valid:
            initial_response_was_valid_json = 0
            json_str = fix_json_with_gpt(json_str, error_msg)

        token_usage = response_json['usage']
        total_input_tokens += token_usage['prompt_tokens']
        total_output_tokens += token_usage['completion_tokens']

        # If the initial JSON was invalid and fixed, we need to account for the token usage of the fix attempt
        if not initial_response_was_valid_json:
            fix_response = call_openai_api_vision(base64_image, "Please fix this JSON.")
            if fix_response.status_code == 200:
                fix_token_usage = fix_response.json()['usage']
                total_input_tokens += fix_token_usage['prompt_tokens']
                total_output_tokens += fix_token_usage['completion_tokens']

        elapsed_time = time.time() - start_time
        result_object = json.loads(json_str)
        result_object.update({
            "time_taken": np.round(elapsed_time,2),
            "initial_response_was_valid_json": initial_response_was_valid_json,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        })
        return result_object
    else:
        return {"error": response.text}




# def chunk_dataframe(df, batch_size):
#     for i in range(0, df.shape[0], batch_size):
#         yield df.iloc[i:i + batch_size]

# def process_batch(batch):
#     results = []
#     for image_batch in batch:
#         # Create a list of tuples (image_ID_index, image_relative_filepath)
#         input_images_list = [(index, filepath) for index, filepath in image_batch]

#         # Instantiate the MultiImage class and combine the images
#         multi_img_instance = MultiImage(input_images_list)

#         # Classify the combined image and store the result
#         classification_json = classify_gpt4v(multi_img_instance.image_combined)
#         results.append(classification_json)

#         # Optionally, save the combined image for debugging
#         combined_image_path = dir_data / "debugging_images" / f"combined_image_{input_images_list[0][0]}.jpg"
#         multi_img_instance.image_combined.save(combined_image_path, "JPEG")

#     return results

# def update_dataframe_with_classifications(df, classifications):
#     # Flatten the list of classification results
#     flat_classifications = [item for sublist in classifications for item in sublist]
#     classification_df = pd.DataFrame(flat_classifications)
#     updated_df = df.merge(classification_df, left_on='image_ID_index', right_on='index', how='left')

#     return updated_df





# def threaded_runner(df, batch_size):
#     # Queue for thread-safe result collection
#     result_queue = Queue()

#     # Executor for running threads
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # Submits batches to the executor
#         futures = [executor.submit(process_batch, batch) for batch in chunk_dataframe(df, batch_size)]

#         # Collects results as they complete
#         for future in as_completed(futures):
#             result_queue.put(future.result())

#     # Collect all results from the queue
#     all_classifications = list(itertools.chain(*list(result_queue.queue)))

#     # Update the original DataFrame with classifications
#     updated_df = update_dataframe_with_classifications(df, all_classifications)

#     return updated_df

# # Example usage:
# df = pd.read_csv(dir_data / "index_image_filepaths.csv", index_col=False).sample(27)
# batch_size = 9
# updated_df = threaded_runner(df, batch_size)
# print(updated_df)

# # Compute and display the confusion matrix
# y_true = df['true_label'].values  # Replace 'true_label' with the appropriate column name
# y_pred = [classification['label'] for classification in all_classifications]  # Extract the label from your JSON object
# compute_and_display_confusion_matrix(y_true, y_pred)




dir_images  = dir_src.parent / "video/frames/"
image_index = create_image_index(dir_images)
image_index.to_csv(dir_data / "index_image_filepaths.csv", index=False)

# image_index_sample = pd.read_csv(dir_data / "index_image_filepaths.csv", index_col=False).head(9)   # Sample first 25
image_index_sample = pd.read_csv(dir_data / "index_image_filepaths.csv", index_col=False).sample(4)   # Sample random 25
print(image_index_sample)

input_images_list = list(zip(image_index_sample['image_ID_index'], image_index_sample['image_relative_filepath']))
multi_img_instance = MultiImage(input_images_list)
multi_img_instance.image_resize_crop_markup(input_images_list[-1][0], input_images_list[-1][1]).save(dir_data / "debugging_images/test.jpg", "JPEG")
multi_img_instance.image_combined.save(dir_data / "debugging_images/test_combined.jpg", "JPEG")

classification_json = classify_gpt4v(multi_img_instance.image_combined)

with open(dir_data / 'debugging_images/classification.json', 'w') as outfile:
    json.dump(classification_json, outfile)















### Stage 1: Image Loading ###
### Producer 1: Reads paths from index_image_filepaths.csv, enqueues them into queue_image_filepaths.
### Consumer 1: Takes image paths from queue_image_filepaths, performs resizing and any required pre-processing, and generates payloads (base64 encoding for GPT4).
### outputs queue_image_filepaths = holds paths to images needing processing.

### Stage 2: Image Batching ###
### Producer 2: Groups payloads from queue_payloads into batches.
### Consumer 2: Forms multi-image batches, stamps images with indexes, enqueues to queue_multiimage_batches.
### outputs queue_multiimage_batches = holds image batches

### Stage 3: API Querying ###
### Producer 3: Takes prepared payloads from queue_multiimage_batches, schedules API calls using futures.
### Consumer 3: Manages API calls, ensuring backpressure handling and error retry logic. Results (API responses) are placed in queue_result_jsons.
### outputs queue_payloads = holds payloads prepared for API querying.

### Stage 4: Result Processing ###
### Producer 4: Monitors queue_result_jsons for completed futures, extracts JSON data from successful API responses.
### Consumer 4: Takes JSON data, validates, and possibly enriches or transforms it. Finalized data is written to disk as immutable checkpoints.
### outputs queue_result_jsons = holds futures/results from API querying.
