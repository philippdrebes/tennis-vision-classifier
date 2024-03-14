

# gpt4v  list of image uri's goes in, classifications go out, classifications are cached

# design idea: checkpoints are saved after each stage, all outputs return their contents so that the functions can be piped.
# Goal: Use 'Command' Design pattern for the queue – it should contain objects who's methods will be executed, rather than just containing data which is passed to seperate executor code

# https://superstudy.guide/algorithms-data-structures/data-structures/stacks-queues/#queues
# gpt4v API documentation:   https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images


from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import base64
import json
import math
import pandas as pd
import re
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
from sklearn.metrics import confusion_matrix
import itertools
import logging
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_log

import sys
sys.path.append('../scripts/')
# from utility_code import profiler_mem_time_detailed    # for bottleneck profiling





logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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






def classify_gpt4v(image):
    # Input: a composite PIL image of frames with index numbers stamped
    # Returns: a JSON of classifications of each frame e.g. {"201": 0, "30": 1, "185": 0, "163": 0, "59": 1, "98": 1, "77": 0, "29": 1, "121": 0, "time_taken": 7.13, "response_state": 1, "cost_tokens_input": 212, "cost_tokens_output": 69}

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

        max_cycles = 3
        attempts_per_cycle = 6  # progression 2, 4, 8, 16, 32, 60
        total_attempts = max_cycles * attempts_per_cycle
        current_attempt = 0

        while current_attempt < total_attempts:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                logging.info(response.json())
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    cycle_position = (current_attempt % attempts_per_cycle) + 1
                    wait = min(2 ** cycle_position, 60)
                    logging.warning(f"{response.json()} Waiting {wait} seconds before retrying...")
                    time.sleep(wait)
                else:
                    logging.error(f"HTTP error occurred: {e} - {response.json()}")
                    return {"error": f"HTTP error occurred: {e}", "response": response.json()}
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error occurred: {e}")
                return {"error": f"Request error occurred: {e}"}

            current_attempt += 1

        return {"error": "Request failed after retries or encountered a non-retryable error."}

    def is_valid_json(json_str):
        try:
            json.loads(json_str)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)

    def fix_json_with_gpt(content, error_message):
        headers = {"Authorization": f"Bearer {API_KEY_OPENAI}"}
        payload = {
            "model": "gpt-3.5-turbo",
            "response_format": "json",
            "messages": [
                {"role": "system", "content": "Fix the errors and return the same content in valid JSON syntax."},
                {"role": "user", "content": f"Error: {error_message}\nContent: {content}"}
            ]
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        logging.info(f"fix_json_with_gpt:\n{response.text}")
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "GPT failed to fix JSON: " + response.text


    time_start = time.time()
    base64_image = encode_image_to_base64(image)
    response = call_openai_api_vision(base64_image)

    response_state = 1
    cost_tokens_input = cost_tokens_output = 0

    logging.info(response)
    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        json_str = re.search(r'{.*}', content, re.DOTALL).group(0) if re.search(r'{.*}', content, re.DOTALL) else '{}'

        is_valid, error_msg = is_valid_json(json_str)
        if not is_valid:
            response_state = 0
            json_str = fix_json_with_gpt(json_str, error_msg)

        token_usage = response_json['usage']
        cost_tokens_input += token_usage['prompt_tokens']
        cost_tokens_output += token_usage['completion_tokens']

        # If the initial JSON was invalid and fixed, we need to account for the token usage of the fix attempt
        if not response_state:
            fix_response = call_openai_api_vision(base64_image, "Please fix this JSON.")
            if fix_response.status_code == 200:
                fix_token_usage = fix_response.json()['usage']
                cost_tokens_input += fix_token_usage['prompt_tokens']
                cost_tokens_output += fix_token_usage['completion_tokens']

        elapsed_time = time.time() - time_start
        response_extracted = json.loads(json_str)
        response_extracted.update({
            "time_taken": np.round(elapsed_time,2),
            "response_state": response_state,
            "cost_tokens_input": cost_tokens_input,
            "cost_tokens_output": cost_tokens_output
        })
        return response_extracted
    else:
        return {"error": response.text}



@dataclass
class MultiImage:
    # Takes in references to images, with their IDs, and composits them into a single labelled image
    input_images_list:  List[Tuple[int, str]]  # List of tuples [(image_ID_index, image_relative_filepath)]
    image_size_default:     int = field(init=True, default=512)
    image_size_shrunk:      int = field(init=True, default=512)
    grid_num_rows_cols:     int = field(init=True, default=1)
    font_size:              int = field(init=True, default=25)
    font_path:              str = field(init=True, default='../scripts/ARIAL.ttf')
    image_combined: Image.Image = field(init=False)

    def __post_init__(self):
        self.grid_num_rows_cols = math.ceil(math.sqrt(len(self.input_images_list)))
        self.image_size_shrunk = math.floor(self.image_size_default / self.grid_num_rows_cols)
        marked_images = [self.image_resize_crop_markup(index, image_filepath) for index, image_filepath in self.input_images_list]
        self.image_combined = self.image_combiner(marked_images)

    def image_resize_crop_markup(self, index, image_filepath):
        img = Image.open(image_filepath)

        # Resizes with constant aspect ratio to height 512
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * self.image_size_default)
        img = img.resize((new_width, self.image_size_default), Image.LANCZOS)

        # Center crop to 512x512
        if new_width > self.image_size_default:
            left = (new_width - self.image_size_default) / 2
            upper = 0
            right = (new_width + self.image_size_default) / 2
            lower = self.image_size_default
            img = img.crop((left, upper, right, lower))

        # Resizes again to fit into a minimal square bounding box
        new_size = self.image_size_shrunk
        img = img.resize((new_size, new_size), Image.LANCZOS)

        # Draws a single pixel wide white line across the top and left borders
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (new_size - 1, 0)], fill="white", width=1)  # Top border
        draw.line([(0, 0), (0, new_size - 1)], fill="white", width=1)  # Left border
        font = ImageFont.truetype(self.font_path, self.font_size)
        text_bbox = draw.textbbox((0, 0), str(index), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        img_width, img_height = img.size
        position = ((img_width - text_width) / 2, (img_height - text_height) / 2)

        # Draws text with slightly offset black border
        offset_range = [-2, 0, 2]
        for x_offset in offset_range:
            for y_offset in offset_range:
                border_position = (position[0] + x_offset, position[1] + y_offset)
                draw.text(border_position, str(index), font=font, fill="black")
        draw.text(position, str(index), font=font, fill="white")
        return img

    def image_combiner(self, marked_images):
        grid_size = self.grid_num_rows_cols
        new_size = self.image_size_shrunk
        canvas = Image.new('RGB', (self.image_size_default, self.image_size_default), "black")
        for i, img in enumerate(marked_images):
            x = (i % grid_size) * new_size
            y = (i // grid_size) * new_size
            canvas.paste(img, (x, y))
        return canvas





@dataclass
class OpenAIAPIPayload:
    # INPUT
    classification_system_message: str
    classification_user_text: str
    classification_image_list: Optional[list] = None
    max_tokens:  int   = 200
    # TEMP VARS
    model:       str   = field(init=False)
    response_format: Optional[dict] = None   # {"type": "json_object"}  or   {"type": "text"}   see https://platform.openai.com/docs/api-reference/chat/create
    temperature: float = 0.0
    messages:    list  = field(init=False)
    # OUTPUT
    payload:     dict  = field(init=False)

    def __post_init__(self):
        self.run()

    @staticmethod
    def encode_image_to_base64(image):
        # Assumes input of a PIL  image.open('image_filepath.jpg') object  , created by MultiImage class
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def create_messages_vision(self):
        self.model = "gpt-4-vision-preview"
        self.messages = [
            {"role": "user", "content": [
                {"type": "text", "text": self.classification_system_message}] +
                [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{OpenAIAPIPayload.encode_image_to_base64(multi_image)}"}} for multi_image in self.classification_image_list]
                        }]

    def create_messages_chat(self):
        self.model = "gpt-3.5-turbo"
        self.response_format = {"type": "json_object"}
        self.messages = [
            {"role": "system", "content": self.classification_system_message},
            {"role": "user", "content": self.classification_user_text}
                        ]

    def run(self):
        isVision = bool(self.classification_image_list)   # If the image list is empty, assumes it is a text query and non_vision
        if isVision:
            self.create_messages_vision()
        else:
            self.create_messages_chat()

        self.payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.messages,
            "max_tokens": self.max_tokens
        }
        if self.response_format:
            self.payload["response_format"] = self.response_format



@dataclass
class OpenAIAPIResponse:
    # INPUT
    payload: dict
    # TEMP VARS
    headers: dict = field(default_factory=lambda: {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY_OPENAI}"})
    response_state: str = 'not_sent'
    time_taken: float = 0
    cost_tokens_input:  int = 0
    cost_tokens_output: int = 0
    cost_total_dollars: float = 0
    # OUTPUT
    response: Optional[dict] = field(init=False)
    response_extracted: Optional[dict] = field(init=False)

    def json_fix_extract_response(self, content):
        try:     # Check if the response content is valid JSON and return extracted response, if not then re-send in another loop to fix it,
            json.loads(content)   # Checks if content contains valid json, should throw JSONDecodeError if not
            json_str = re.search(r'{.*}', content, re.DOTALL).group(0)

            if self.response_state == 'fixing':
                self.response_state = 'fixed'
            else:
                self.response_state = 'received_no_errors'
            response_extracted = json.loads(json_str)
            response_extracted.update({
                "time_taken": self.time_taken,
                "response_state": self.response_state,
                "cost_tokens_input": self.cost_tokens_input,
                "cost_tokens_output": self.cost_tokens_output,
                "cost_total_dollars": self.cost_total_dollars })

            self.response_extracted = response_extracted
            logging.info(f'json_fix_extract_response response_extracted:\n{self.response_extracted}')

        except json.JSONDecodeError as e:
            logging.warning(f'JSON format error:\n{e}')
            if self.response_state == 'fixing':   # If it already tried once, give up
               self.response_extracted = {
                "response_state": 'fixing_failed',
                "error": "JSON couldn't be fixed despite trying, check character limit or format.",
                "content": content }
            else:   # Sends it for fixing using another request
                self.response_state = 'fixing'
                payload_fix_json = OpenAIAPIPayload(
                classification_system_message='Fix any syntax errors and return the same content in valid JSON syntax with no newlines. If any fields are truncated or incomplete, drop them.',
                classification_user_text=content,
                classification_image_list=None,  # No images, non-vision request
                max_tokens = int(self.payload['max_tokens'] * 2)   # Increase the token limit in case the problem was truncation
                    )
                logging.error(f'number of tokens in fixer payload: {payload_fix_json.max_tokens}')
                response_fix_json = OpenAIAPIResponse(payload=payload_fix_json.payload)
                response_fix_json.handle_openai_request()

                if response_fix_json.response_state == 'received_no_errors':
                    self.response_extracted = response_fix_json.response_extracted  #The fixing worked
                else:   # The fixing didn't work
                    self.response_extracted = {
                        "response_state": 'fixing_failed',
                        "error": "Attempted to fix JSON but failed, even after x2 the max_tokens.",
                        "content": response_fix_json.response }

        except Exception as e:
            logging.warning(f'json_fix_extract_response unexpected error:\n{e}')
            self.response_extracted = str(e)


    @retry(wait=wait_exponential(min=5, max=30), stop=stop_after_attempt(2), retry=retry_if_exception_type(requests.exceptions.HTTPError), before=before_log(logging.getLogger(__name__), logging.DEBUG))
    def handle_openai_request(self):
        time_start = time.time()
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=self.payload)
            response.raise_for_status()  # This will raise an HTTPError for bad responses

            self.response = response.json()
            self.time_taken += np.round(time.time() - time_start,2)
            self.cost_tokens_input  += self.response.get('usage', {}).get('prompt_tokens', 0)
            self.cost_tokens_output += self.response.get('usage', {}).get('completion_tokens', 0)
            self.cost_total_dollars = np.round((30*(self.cost_tokens_input / 1000000)),4) + np.round((30*(self.cost_tokens_output / 1000000)),4)   # Assuming GPT4 prices:   https://openai.com/pricing

            logging.info('handle_openai_request response Received')
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            self.json_fix_extract_response(content)   # This will handle any JSON syntax errors and always specify response_extracted
            logging.info(f'handle_openai_request response_extracted:\n{self.response_extracted}')

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.error("Rate limit exceeded.")
                self.response_extracted = {
                    "response_state": False,
                    "error": f"Rate limit exceeded: HTTP error: {e.response.status_code} - {e.response.reason}",
                    "content": e.response.text if e.response is not None else "No response text due to rate limiting." }
            else:
                logging.error(f"HTTP Error encountered: {e.response.status_code} - {e.response.reason}")
                self.response_extracted = {
                    "response_state": False,
                    "error": f"HTTP error: {e.response.status_code} - {e.response.reason}",
                    "content": e.response.text if e.response is not None else "No response text due to HTTP error." }

        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            self.response_extracted = {
                "response_state": False,
                "error": f"Unexpected error: {str(e)}",
                "content": "An unexpected error prevented the request from completing." }



# @profiler_mem_time_detailed
def classify_gpt4v_threaded(batch_size:int, read_in_head_limit, images_index_input, images_index_output):
    #Input: df_images_index_input index of image filepaths, and batch iteration size
    #Output: df_images_output index containing the gpt4v classifications


    def setup_input_output_indexes(read_in_head_limit, images_index_input, images_index_output):
        df_images_index_input = pd.read_csv(images_index_input, index_col=False, nrows=read_in_head_limit)
        if 'classification_gpt4v' not in df_images_index_input.columns:
            df_images_index_input['classification_gpt4v'] = np.nan

        if not images_index_output.exists():
            logging.info(f"Output index   {images_index_output.name}   is missing, creating it.")
            df_images_index_input = df_images_index_input[[col for col in df_images_index_input.columns if col != 'image_relative_filepath'] + ['image_relative_filepath']]
            df_images_index_input.to_csv(images_index_output, index=False)
            df_images_index = df_images_index_input.copy()
        else:
            df_images_index = pd.read_csv(images_index_output, index_col=False)
        return df_images_index

    def chunk_dataframe(df_images_index_todo, batch_size):
        for i in range(0, df_images_index_todo.shape[0], batch_size):
            yield df_images_index_todo.iloc[i:i + batch_size]

    def process_batch(batch):
        # Composits the combined image, sends it off for classifying, and stores the result
        results = []
        input_images_list = []
        for _, row in batch.iterrows():
            input_images_list.append((row['image_ID_index'], row['image_relative_filepath']))

        multi_img_instance = MultiImage(input_images_list)
        # classification_dict = classify_gpt4v(multi_img_instance.image_combined)   # Deprecated

        payload_classify = OpenAIAPIPayload(
                classification_system_message="Classify each sub-frame as PLAYING (with label 1) or NON-PLAYING (with label 0): PLAYING scenes only show the tennis court from the standard broadcast angle, fully visible and vertically & horizontally aligned with the court lines (the audience should at most be barely visible). NON-PLAYING scenes show any other content that is not a standard broadcast view of the court, including low-angle, court-side, zoom-in, or audience shots. Respond just in a JSON using the centered number (white text with black outline) in each square as the index for each frame.",
                classification_user_text="Classify each sub-frame as PLAYING (with label 1) or NON-PLAYING (with label 0): PLAYING scenes only show the tennis court from the standard broadcast angle, fully visible and vertically & horizontally aligned with the court lines (the audience should at most be barely visible). NON-PLAYING scenes show any other content that is not a standard broadcast view of the court, including low-angle, court-side, zoom-in, or audience shots. Respond just in a JSON using the centered number (white text with black outline) in each square as the index for each frame.",
                classification_image_list=[multi_img_instance.image_combined],
                max_tokens = 400   # Increase the token limit in case the problem was truncation
                    )
        # logging.info(f'payload_classify:\n{payload_classify.payload}')
        response_handler = OpenAIAPIResponse(payload=payload_classify.payload)
        response_handler.handle_openai_request()
        classification_dict = response_handler.response_extracted
        results.append(classification_dict)
        logging.info(f'classification_dict:\n{classification_dict}')
        combined_image_path = dir_data / "debugging_images" / f"combined_image_{input_images_list[0][0]}.jpg"   # Optional, for debugging
        multi_img_instance.image_combined.save(combined_image_path, "JPEG")
        return results

    write_lock = Lock()

    def thread_safe_update_concurrent_and_write_to_output(df_update, df_concurrent, output_file):
        with write_lock:
            df_concurrent = merge_df_using__image_ID_index_(df_concurrent, df_update)
            df_concurrent['classification_gpt4v'] = df_concurrent['classification_gpt4v'].apply(
                lambda x: '{:.0f}'.format(float(x)) if not pd.isnull(x) and isinstance(x, (float, int)) else x)
            df_concurrent.to_csv(output_file, index=False)
        return df_concurrent

    def merge_df_using__image_ID_index_(df_base, df_update):
        # logging.info(f'merge_df_using__image_ID_index_:\n{df_base}\n{df_update}')

        df_update = df_update.drop_duplicates(subset=['image_ID_index'], keep='first')
        df_update.set_index(['image_ID_index'], inplace=True)
        df_base.set_index(['image_ID_index'], inplace=True)
        df_base.update(df_update)
        df_base.reset_index(inplace=True)
        return df_base

    def update_dataframe_with_classifications(df_images_index_todo, classifications):
        classifications = [classification for classification in classifications if 'response_state' in classification]   # removes any dicts that are missing the 'response_state' field i.e. which means they're are failed runs
        flattened_classification_results = {}
        for classification in classifications:
            for key, value in classification.items():
                if key.isdigit():  # This selects imageIDs assuming they are always the only keys that are integers
                    flattened_classification_results[int(key)] = value
        logging.info(f'flattened_classification_results:\n{flattened_classification_results}')
        df_classifications = pd.DataFrame(list(flattened_classification_results.items()), columns=['image_ID_index', 'classification_gpt4v'])
        df_classifications['classification_gpt4v'] = df_classifications['classification_gpt4v'].astype(int)

        logging.info(f'df_classifications:\n{df_classifications}')
        merge_df_using__image_ID_index_(df_images_index_todo, df_classifications)
        logging.info(f'df_images_index_processed:\n{df_images_index_todo}')

        # df = df_images_index_todo.merge(df_classifications, on='image_ID_index', how='left')   # This was removed because it caused _x _y conflicts
        return df_images_index_todo[[col for col in df_images_index_todo.columns if col != 'image_relative_filepath'] + ['image_relative_filepath']]

    def run_logging_to_file(classifications_merged):
        logging.info(f'classifications_merged:\n{classifications_merged}')
        classifications_merged = [classification for classification in classifications_merged if 'response_state' in classification]   # removes any dicts that are missing the 'response_state' field i.e. that are failed runs
        # run_cost_tokens_input = [sum([classification['cost_tokens_input'] for classification in classifications_merged])]
        # run_cost_tokens_output = [sum([classification['cost_tokens_output'] for classification in classifications_merged])]
        # run_response_states = [classification['response_state'] for classification in classifications_merged]
        # run_estimated_total_price_dollars = [np.round((30*(run_cost_tokens_input[0] / 1000000)),2) + np.round((30*(run_cost_tokens_output[0] / 1000000)),2)]
        run_logging = pd.DataFrame({
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            # 'run_cost_tokens_input': run_cost_tokens_input,
            # 'run_cost_tokens_output': run_cost_tokens_output,
            # 'run_response_states': run_response_states,   # TODO
            # 'run_estimated_total_price_dollars': run_estimated_total_price_dollars,   # Assuming tokens only spent on gpt4 vision endpoint:   https://openai.com/pricing
            })
        logging.info(f'classifications_merged:\n{classifications_merged}')
        if not (dir_data / "run_logging.csv").exists():
            run_logging.to_csv(dir_data / "run_logging.csv", index=False)
        else:
            run_logging_csv = pd.read_csv(dir_data / "run_logging.csv", index_col=False)
            run_logging_csv_merged = pd.concat([run_logging_csv, run_logging], ignore_index=True)
            run_logging_csv_merged.to_csv(dir_data / "run_logging.csv", index=False)


    df_images_index = setup_input_output_indexes(read_in_head_limit, images_index_input, images_index_output)
    df_images_index_todo = df_images_index[df_images_index['classification_gpt4v'].isna()]   # Drops all finished lines
    logging.info(f'Index of ALL images:\n{df_images_index}')
    logging.info(f'Index of UNPROCESSED images:\n{df_images_index_todo}')
    df_images_index_concurrent = df_images_index.copy()


    result_queue = Queue()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for batch in chunk_dataframe(df_images_index_todo, batch_size):
            ids = batch['image_ID_index'].tolist()
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Enqueueing batch of with image IDs: {' '.join(map(str, ids))}")
            future = executor.submit(process_batch, batch)
            futures.append(future)

    # Simple Batched method
        # for future in as_completed(futures):
        #     result_queue.put(future.result())
    # df_images_index_processed = update_dataframe_with_classifications(df_images_index_todo, classifications_merged)
    # df_images_output = merge_df_using__image_ID_index_(df_images_index.copy(), df_images_index_processed)
    # logging.info(f'df_images_output:\n{df_images_output}')
    # df_images_output.to_csv(images_index_output, index=False)

        for future in as_completed(futures):
            df_images_index_processed  = update_dataframe_with_classifications(df_images_index_todo, future.result())
            df_images_index_concurrent = thread_safe_update_concurrent_and_write_to_output(df_images_index_processed, df_images_index_concurrent, images_index_output)
            result_queue.put(future.result())

    df_images_output = df_images_index_concurrent

    classifications_merged = list(itertools.chain(*list(result_queue.queue)))
    run_logging_to_file(classifications_merged)

    return df_images_output












## Example usage code for classify_gpt4v_threaded(image) ###

# batch_size = 9
# read_in_head_limit = None   # set to None to read in all values
# images_index_input  = dir_data / "index_image_filepaths.csv"
# images_index_output = dir_data / "index_image_filepaths_classifications_gpt4v.csv"
# updated_df = classify_gpt4v_threaded(batch_size, read_in_head_limit, images_index_input, images_index_output)







### TODO ###
# ☐ confusion dash
# ☐ Test resizing the images to square without cropping
# ☐ Test how small you can go, putting the image labels into the text
# ☐ Test providing a 'guide' image for edge case comprehension
# ☐ Test using 'Histograms' in image classification






# # Compute and display the confusion matrix
# y_true = df['true_label'].values  # Replace 'true_label' with the appropriate column name
# y_pred = [classification['label'] for classification in classifications_merged]  # Extract the label from your JSON object
# compute_and_display_confusion_matrix(y_true, y_pred)







# Tips for using the OpenAI Vision API:
    # OpenAI Vision API Reference:       Supports multiple images at once   https://platform.openai.com/docs/guides/vision/multiple-image-inputs
    # Should we use the OpenAI package:  I prefer not to, and just send requests using import requests   because the OpenAI package is very frequently updated and often gives dependency conflict issues
    # Rate Limit Requests Per Day: Batch miltiple images into a single request
    # Rate Limit max_tokens:       It is calculate based on the max_tokens you request, not your actual usage, so keep your request as low as you can
    # Rate Limit Countdown:        Check how far-off you are from getting rate
    # Minimise Token Usage:        use  detail: low and 512x512 images where possible to minimise costs
    # Utility as a 0-shot classifier:        Very fast & convenient. Means you can skip training your own model, or orchestrating your own GPU environment (train, or production)
    # Reliability as a 0-shot classifier:    Varies, it depends, but is pretty good. Consider using it to pre-label your
