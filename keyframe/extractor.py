import cv2
import numpy as np
import get_supers
import object_track
import concurrent.futures


def getContrast(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return np.std(gray) + (0.3 * np.mean(gray)), np.mean(gray)


def getColour(f):
    # based on example from the following link
    # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    (B, G, R) = cv2.split(f.astype("float"))
    rg = abs(R - G)
    yb = abs(0.5 * (R + G) - B)
    stdRoot = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    meanRoot = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return stdRoot + (0.3 * meanRoot), (np.mean(B), np.mean(G), np.mean(R))


def getEdge(f):
    lap = cv2.Laplacian(f, cv2.CV_64F)
    return np.std(lap) ** 2, np.mean(lap)


def getSupQual(quality, supers):
    supQual = []
    for i in range(len(supers) - 1):
        supQual.append(np.average(quality[supers[i]:supers[i + 1]]))
    return supQual


def getTime(frame, videoFile):
    cap = cv2.VideoCapture(videoFile)
    fps = (cap.get(5))
    return (int)(frame / fps)


def processVideo(videoFile, start_frame_number):
    cap = cv2.VideoCapture(videoFile)
    fps = int(cap.get(5))
    print('Number of Frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion = []
    camMot = []
    edges = []
    uEdges = []
    colours = []
    uColours = []
    contrasts = []
    uContrasts = []
    unique = np.array([0, 0, 0])

    print("Calculating motion, and quality...")

    # calculate motion
    motion.append(0)
    flow = None
    # grab first frame since missing form loop
    ret, old = cap.read()
    while True:
        if old.shape[0] > 480 and old.shape[1] > 640:
            old = cv2.pyrDown(old)
        else:
            break
    # old = cv2.resize(old,(640, 480),interpolation=0)
    t1, t2 = getEdge(old)  # creates temp variables
    edges.append(t1)
    uEdges.append(t2)
    t1, t2 = getColour(old)
    colours.append(t1)
    uColours.append(t2)
    t1, t2 = getContrast(old)
    contrasts.append(t1)
    uContrasts.append(t2)

    i = 1
    while (True):
        j = 0
        while (j < start_frame_number):
            ret, new = cap.read()
            j = j + 1
        if not ret:
            break
        while True:
            if new.shape[0] > 480 and new.shape[1] > 640:
                new = cv2.pyrDown(new)
            else:
                break
        # new = cv2.resize(new,(640, 480),interpolation=2)
        t1, t2 = getEdge(new)  # creates temp variables
        edges.append(t1)
        uEdges.append(t2)
        t1, t2 = getColour(new)
        colours.append(t1)
        uColours.append(t2)
        t1, t2 = getContrast(new)
        contrasts.append(t1)
        uContrasts.append(t2)

        gray1 = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray2, gray1, flow, 0.5, 2, 15, 2, 5, 1.1,
                                            1)  # calculate the optical flow of entire frame
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.ma.masked_invalid(mag)  # prevent errors from inf values
        motion.append(np.mean(mag))
        camMot.append(np.median(mag))

        if i % 100 == 0:
            print("Frame " + str(i * start_frame_number) + ": " + str(motion[i]))

        old = new
        i += 1

    cap.release()
    # normalize motion
    motion = np.array(motion)
    motion = motion / max(motion)

    # normalize quality
    mEdge = np.mean(edges)
    mCol = np.mean(colours)
    mCon = np.mean(contrasts)
    quality = []
    for i in range(len(edges)):
        quality.append(edges[i] / mEdge + colours[i] / mCol + contrasts[i] / mCon)

    # added motion smoothing here instead of having seperate file
    motion = np.convolve(motion, [0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05], "same")  # seudo gaussian

    print("Calculating superframes...")

    supers = get_supers.getSupers(motion, fps)

    print("Calculating superframe quality...")

    supQual = getSupQual(quality, supers)
    # calculate uniqueness, edges colours contrasts
    unique = []
    uMEdge = np.mean(uEdges)
    uMCol = np.mean(uColours)
    uMCon = np.mean(uContrasts)
    for i in range(len(supers) - 1):
        unique.append(abs(np.mean(uEdges[supers[i]:supers[i + 1]]) - uMEdge) +
                      abs(np.linalg.norm(np.mean(uColours[supers[i]:supers[i + 1]]) - uMCol)) +
                      abs(np.mean(uContrasts[supers[i]:supers[i + 1]]) - uMCon))
    unique = np.array(unique) / max(unique)

    print("Calculating object tracking...")

    supQual = np.array(supQual)
    supQual = supQual / max(supQual)

    scores = object_track.getTrack(supers, camMot, videoFile, supQual)

    # normalizing
    scores = np.array(scores)
    scores = scores / max(scores)
    lens = []
    for i in range(len(supers) - 1):
        lens.append(supers[i + 1] - supers[i])

    return supQual, scores, lens, motion, unique, supers


def process_frame(index, frame, start_frame_number):
    if frame.shape[0] > 480 and frame.shape[1] > 640:
        frame = cv2.pyrDown(frame)

    t1, t2 = getEdge(frame)
    edges = t1
    uEdges = t2
    t1, t2 = getColour(frame)
    colours = t1
    uColours = t2
    t1, t2 = getContrast(frame)
    contrasts = t1
    uContrasts = t2

    return index, edges, uEdges, colours, uColours, contrasts, uContrasts


def processVideoP(videoFile, start_frame_number):
    cap = cv2.VideoCapture(videoFile)
    fps = int(cap.get(5))
    print('Number of Frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion = []
    camMot = []
    edges = []
    uEdges = []
    colours = []
    uColours = []
    contrasts = []
    uContrasts = []
    unique = np.array([0, 0, 0])

    print("Calculating motion, and quality...")

    # calculate motion
    motion.append(0)
    flow = None
    ret, old = cap.read()

    # Parallel processing of frames
    with concurrent.futures.ThreadPoolExecutor() as executor:
        frame_results = []

        # Submit a future for the first frame
        if ret:
            future = executor.submit(process_frame, 0, old, start_frame_number)
            frame_results.append(future)

        # Process remaining frames
        i = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames until reaching the start frame number
            if i < start_frame_number:
                i += 1
                continue

            future = executor.submit(process_frame, i, frame, start_frame_number)
            frame_results.append(future)
            i += 1

        # Collecting results
        for future in concurrent.futures.as_completed(frame_results):
            index, t1, t2, t3, t4, t5, t6 = future.result()
            edges.append(t1)
            uEdges.append(t2)
            colours.append(t3)
            uColours.append(t4)
            contrasts.append(t5)
            uContrasts.append(t6)

    cap.release()

    # normalize motion
    motion = np.array(motion)
    motion = motion / max(motion)

    # normalize quality
    mEdge = np.mean(edges)
    mCol = np.mean(colours)
    mCon = np.mean(contrasts)
    quality = []
    for i in range(len(edges)):
        quality.append(edges[i] / mEdge + colours[i] / mCol + contrasts[i] / mCon)

    # added motion smoothing here instead of having seperate file
    motion = np.convolve(motion, [0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05], "same")  # seudo gaussian

    print("Calculating superframes...")

    supers = get_supers.getSupers(motion, fps)

    print("Calculating superframe quality...")

    supQual = getSupQual(quality, supers)
    # calculate uniqueness, edges colours contrasts
    unique = []
    uMEdge = np.mean(uEdges)
    uMCol = np.mean(uColours)
    uMCon = np.mean(uContrasts)
    for i in range(len(supers) - 1):
        unique.append(abs(np.mean(uEdges[supers[i]:supers[i + 1]]) - uMEdge) +
                      abs(np.linalg.norm(np.mean(uColours[supers[i]:supers[i + 1]]) - uMCol)) +
                      abs(np.mean(uContrasts[supers[i]:supers[i + 1]]) - uMCon))
    unique = np.array(unique) / max(unique)

    print("Calculating object tracking...")

    supQual = np.array(supQual)
    supQual = supQual / max(supQual)

    scores = object_track.getTrack(supers, camMot, videoFile, supQual)

    # normalizing
    scores = np.array(scores)
    scores = scores / max(scores)
    lens = []
    for i in range(len(supers) - 1):
        lens.append(supers[i + 1] - supers[i])

    return supQual, scores, lens, motion, unique, supers


supQual, scores, lens, motion, unique, supers = processVideo(
    "../video/Roger Federer vs Rafael Nadal ｜ Wimbledon 2019 ｜ Full Match [wZnCcqm_g-E].mp4", 1)

segments = [idx for idx, val in enumerate(scores) if val > 0.12]
timeStamp_segments = [idx for idx, val in enumerate(scores) if val > 0.20]
print(segments)
print(timeStamp_segments)

timeStamps = []
frameSegments = []
for i in segments:
    start = getTime(supers[i], "../video/Roger Federer vs Rafael Nadal ｜ Wimbledon 2019 ｜ Full Match [wZnCcqm_g-E].mp4")
    end = getTime(supers[i + 1],
                  "../video/Roger Federer vs Rafael Nadal ｜ Wimbledon 2019 ｜ Full Match [wZnCcqm_g-E].mp4")
    timeStamps.append((start, end))
    frameSegments.append((supers[i], supers[i + 1]))

timeStampsForTimeStamp = []
frameSegmentsForTimeStamp = []
for i in timeStamp_segments:
    start = getTime(supers[i], "../video/Roger Federer vs Rafael Nadal ｜ Wimbledon 2019 ｜ Full Match [wZnCcqm_g-E].mp4")
    end = getTime(supers[i + 1],
                  "../video/Roger Federer vs Rafael Nadal ｜ Wimbledon 2019 ｜ Full Match [wZnCcqm_g-E].mp4")
    timeStampsForTimeStamp.append((start, end))
    frameSegmentsForTimeStamp.append((supers[i], supers[i + 1]))
print(frameSegments)
