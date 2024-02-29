


### GET FRAMES ###

# ffmpeg -ss 00:00:00 -i "Maria Sharapova vs Caroline Wozniacki Full Match ｜ US Open 2014 Round Four [72VhC9biEFk].mp4" -t 00:15:00 -c copy "test.mp4"

# # use videoprocessor  cap.read()  if you want live read-in,   or retrofit  extract_frames.sh

# 	#!/bin/bash

# 	# Directory containing your video files
# 	VIDEO_DIR="../video"

# 	# Directory where you want to save the extracted frames
# 	OUTPUT_DIR="../video/frames"

# 	# Create the output directory if it doesn't exist
# 	mkdir -p "$OUTPUT_DIR"

# 	# Iterate over each video file in the video directory
# 	for video in "$VIDEO_DIR"/*.mp4; do
# 	  VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/$(basename "$video" .mp4)"

# 	  # Check if VIDEO_OUTPUT_DIR already exists
# 	  if [ -d "$VIDEO_OUTPUT_DIR" ]; then
# 	    echo "Directory $VIDEO_OUTPUT_DIR already exists. Skipping..."
# 	    continue # Skip to the next video file
# 	  fi

# 	  mkdir -p "$VIDEO_OUTPUT_DIR"
# 	  # Use FFmpeg to extract one frame per second
# 	  ffmpeg -i "$video" -vf "fps=1" "$VIDEO_OUTPUT_DIR/$(basename "$video" .mp4)_frame_%04d.jpg"
# 	done






### FRAME DIFFERENCING ###
# test 'neighbors are similar' metrics for a 'timeline-consistency check'
# test out techniques to identify if adjacent frames in a rolling average are of the same 'scene'
#  Scale-Invariant Feature Transform (SIFT), Speeded Up Robust Features (SURF), or ORB (Oriented FAST and Rotated BRIEF).
# before 🟩🟩🟩🟩🟩🟩🟩🟦🟦🟦🟦🟦🟦🟦🟩🟩🟩🟩🟩🟩🟩🟩
# after  🟩🟩🟩🟩🟩🟩🟩🟥🟦🟦🟦🟦🟦🟦🟥🟩🟩🟩🟩🟩🟩🟩











# Let's conceptualize the timeline of a tennis broadcast video stream using block emojis to represent different shots. We'll use different colors to denote distinct types of shots/scenes (e.g., play, waiting, audience shots, close-ups). The rolling window concept with a size of 5 frames will be illustrated to help identify change points effectively.

# ### Timeline Representation

# - 🟩: Play (balanced stable shot of the whole court)
# - 🟦: Waiting (any other scenario not showing the court, like player close-ups, audience, etc.)
# - 🟥: Significant change points (where the broadcast switches between shots)

# ### Example Stream of Frames:

# ```
# 🟩🟩🟩🟩🟩🟩🟩🟩🟦 🟦🟦🟦🟦🟦🟦🟦🟩🟩🟩🟩🟩🟩🟩🟩


# new 🟩🟩🟩🟩🟩🟩🟩🟩🟦🟦🟦🟩🟦🟩🟦🟦🟩🟩🟩🟩🟩🟩🟩🟩
# ```


# 🟩🟦🟩🟦🟦



# ### Identifying Change Points with a Sliding Window:

# 1. **Initial Window (all play):**

#    ```
#    🟩🟩🟩🟩🟩
#    ```

# 2. **Sliding the Window (still no change):**

#    ```
#    🟩🟩🟩🟩🟩 (next frame is similar, continue)
#    ```

# 3. **Approaching a Change Point (transition from play to waiting):**

#    ```
#    🟩🟩🟩🟩🟦 (last frame starts to change, nearing change point)
#    ```

# 4. **At a Change Point (identified by a significant difference within the window):**

#    ```
#    🟩🟩🟩🟦🟦 (significant change noted, 🟥 could represent this point)
#    ```

# 5. **Post Change Point (new sequence of waiting frames begins):**

#    ```
#    🟦🟦🟦🟦🟦 (now fully in a new shot type, continue until next change)
#    ```

# 6. **Next Change Point (transition back to play):**

#    ```
#    🟦🟦🟦🟦🟩 (change point approaching, prepare to mark)
#    ```

# ### Conceptual Overview with Change Points Marked:

# ```
# 🟩🟩🟩🟩🟩🟩🟩🟥🟦🟦🟦🟦🟦🟦🟦🟥🟩🟩🟩🟩🟩🟩🟩🟩
# ```

# ### Strategy for System Design:
# - **Monitor Frame Similarity:** Use the rolling window of 5 frames to monitor similarity. When the similarity within this window drops (indicating a transition from play to waiting or vice versa), mark this as a potential change point.
# - **Marking Change Points:** Once a change point is identified (e.g., where the type of shot significantly changes), mark this frame. You might choose to analyze the frame right before the change and the first frame of the new shot type more closely.
# - **Selecting Representative Frames:** From each identified shot type (chunk), select up to 4 representative frames for detailed analysis. Ideally, these would include the frame at the change point and subsequent frames to ensure accurate identification of the shot type.
# This approach enables efficient processing by focusing on frames that represent potential changes in the broadcast, thus minimizing the number of frames sent for detailed analysis while ensuring accurate shot classification.
