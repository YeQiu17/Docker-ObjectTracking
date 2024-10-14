# main.py

import sys
import time
import cv2
import numpy as np
import torch
import logging
from scipy.spatial import distance
from munkres import Munkres
from yolov5 import YOLOv5 
from torchvision import transforms
from osnet import osnet_x1_0  # Torchreid OSNet model import
from multiprocessing import Pool, cpu_count, Manager
import math
from db2 import save_feature_to_db, get_features_from_db, update_feature_in_db, get_unique_people_count
import random
import traceback

class Object:
    def __init__(self, pos, feature=None, id=-1):
        self.feature = feature
        self.id = id
        self.time = time.monotonic()
        self.pos = pos
        self.feature_history = [feature] if feature is not None else []
        self.partial_time = time.monotonic()

    def update_feature(self, new_feature):
        """Update the object feature vector using a running average."""
        self.feature_history.append(new_feature)
        if len(self.feature_history) > 10:  # Limit history to 10
            self.feature_history.pop(0)
        self.feature = np.mean(self.feature_history, axis=0)

    def is_fully_visible(self, frame_width, frame_height):
        """Check if the object is fully visible in the frame."""
        xmin, ymin, xmax, ymax = self.pos
        return xmin >= 0 and ymin >= 0 and xmax <= frame_width and ymax <= frame_height


# Model paths
yolo_model_path = 'models/yolov5s.pt'
reid_model_path = 'models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
video_capture_list = ['cam-1M.mp4', 'cam-2M.mp4']
# video_capture_list = [r'rtsp://yectra:Yectra123@192.168.1.59:554/stream1']
# video_capture_list = ['campus4-c0.mp4', 'campus4-c1.mp4', 'campus4-c2.mp4']
# video_capture_list = ['video5.mp4']
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def load_reid_model(model_path):
    """Load OSNet ReID model, ignoring classifier layer mismatch."""
    model = osnet_x1_0(pretrained=False)  # Load OSNet without pre-trained classifier
    
    # Remove the classifier layer as it is not needed for feature extraction
    if hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()  # Replace with an identity layer
    
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)  # Ignore mismatches in classifier
        logging.debug("OSNet ReID model loaded successfully, classifier layer ignored.")
    except Exception as e:
        logging.error(f"Error loading OSNet model: {e}")
        raise

    model.eval()  # Set model to evaluation mode
    return model

def extract_features(model, img, device):
    """Extract feature vectors from the detected object image."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    logging.debug(f"Features extracted: {features.cpu().numpy().flatten()[:10]}...")
    return features.cpu().numpy().flatten()

def combine_frames(frames):
    """Combine frames from multiple cameras into one for visualization."""
    num_frames = len(frames)
    if num_frames == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    grid_size = math.ceil(math.sqrt(num_frames))
    frame_height, frame_width, _ = frames[0].shape
    combined_frame = np.zeros((grid_size * frame_height, grid_size * frame_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r = idx // grid_size
        c = idx % grid_size
        combined_frame[r * frame_height:(r + 1) * frame_height, c * frame_width:(c + 1) * frame_width] = frame

    return combined_frame

def check_opencv_gui_support():
    """Check if the system supports OpenCV GUI functionality."""
    try:
        cv2.imshow('test', np.zeros((1, 1), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False

def process_frame(args):
    """Process each frame to detect objects and extract features."""
    yolo, reid_model, frame, device, partial_objects, partial_timeout = args
    objects = []
    results = yolo.predict(frame)

    frame_height, frame_width, _ = frame.shape
    logging.debug(f"Detections: {len(results.xyxy[0])} objects detected.")

    # Remove stale partially visible objects
    current_time = time.monotonic()
    partial_objects[:] = [obj for obj in partial_objects if current_time - obj.partial_time <= partial_timeout]

    for det in results.xyxy[0].cpu().numpy():
        xmin, ymin, xmax, ymax, conf, cls = det
        if conf > 0.6 and int(cls) == 0:  # Only consider person class with confidence > 0.6
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

            detected_object = Object([xmin, ymin, xmax, ymax])

            # Check if the object is fully visible
            if detected_object.is_fully_visible(frame_width, frame_height):
                obj_img = frame[ymin:ymax, xmin:xmax]
                vec = extract_features(reid_model, obj_img, device)
                detected_object.feature = vec
                objects.append(detected_object)
                logging.debug(f"Object fully visible at [{xmin}, {ymin}, {xmax}, {ymax}] with confidence {conf}.")
            else:
                partial_objects.append(detected_object)
                logging.debug(f"Object partially visible at [{xmin}, {ymin}, {xmax}, {ymax}].")

    # Check previously partially visible objects to see if they are now fully visible
    fully_visible_objects = []
    for obj in partial_objects:
        if obj.is_fully_visible(frame_width, frame_height):
            obj_img = frame[obj.pos[1]:obj.pos[3], obj.pos[0]:obj.pos[2]]
            vec = extract_features(reid_model, obj_img, device)
            obj.feature = vec
            fully_visible_objects.append(obj)
            logging.debug(f"Previously partial object is now fully visible at {obj.pos}.")

    # Remove fully visible objects from the partial_objects list
    for obj in fully_visible_objects:
        partial_objects.remove(obj)

    # Add fully visible objects from partial list to current frame objects
    objects.extend(fully_visible_objects)

    return objects

def initialize_video_captures(video_capture_list):
    """Initialize video capture objects for multiple cameras."""
    caps = [cv2.VideoCapture(vcap) for vcap in video_capture_list]
    return caps

def release_video_captures(caps):
    """Release all video capture objects."""
    for cap in caps:
        cap.release()

def generate_color_from_id(obj_id):
    """Generate a unique color for each object ID using a hash function."""
    random.seed(hash(obj_id))  # Use hash of the string ID as seed for random color
    return tuple([random.randint(0, 255) for _ in range(3)])

def main():
    # Persistent ID count logic
    unique_ids = set()
    id_num = get_unique_people_count()  # Start ID count from database
    logging.info(f"Starting ID count from database: {id_num}")

    dist_threshold = 0.5
    timeout_threshold = 5
    partial_timeout = 2
    frame_count = 0
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLOv5(yolo_model_path, device=device)
    reid_model = load_reid_model(reid_model_path).to(device)

    caps = initialize_video_captures(video_capture_list)
    gui_supported = check_opencv_gui_support()

    with Manager() as manager:
        partial_objects_list = manager.list([[] for _ in caps])

        try:
            with Pool(cpu_count()) as pool:
                while True:
                    if gui_supported and cv2.waitKey(1) == 27:
                        logging.info("Exit signal received. Exiting...")
                        break

                    frames = [cap.read()[1] for cap in caps]
                    if any(frame is None for frame in frames):
                        logging.warning("One or more video streams ended. Exiting...")
                        break

                    args = [(yolo, reid_model, frame, device, partial_objects_list[cam_idx], partial_timeout)
                            for cam_idx, frame in enumerate(frames)]
                    objects_list = pool.map(process_frame, args)

                    total_objects = sum(len(objects) for objects in objects_list)
                    logging.debug(f"Total objects detected across all cameras: {total_objects}")
                    if total_objects == 0:
                        if gui_supported:
                            for i in range(len(caps)):
                                cv2.imshow(f'cam{i}', frames[i])
                        continue

                    # Retrieve features from the database
                    features_db = get_features_from_db()

                    hungarian = Munkres()
                    for cam in range(len(caps)):
                        if not features_db or not objects_list[cam]:
                            continue

                        dist_matrix = []
                        for obj_cam in objects_list[cam]:
                            db_features = []
                            for obj_id, feature in features_db.items():
                                db_features.append(feature)

                            dist_matrix.append([distance.cosine(obj_db, obj_cam.feature) for obj_db in db_features])

                        combination = hungarian.compute(dist_matrix)

                        for idx_obj, idx_db in combination:
                            if objects_list[cam][idx_obj].id != -1:
                                continue
                            
                            matched_obj_id = list(features_db.keys())[idx_db]
                            matched_feature = features_db[matched_obj_id]
                            dist = distance.cosine(objects_list[cam][idx_obj].feature, matched_feature)

                            if dist < dist_threshold:
                                logging.debug(f"Matching object {objects_list[cam][idx_obj].id} with DB object {matched_obj_id} (distance: {dist:.4f}).")
                                update_feature_in_db(matched_obj_id, objects_list[cam][idx_obj].feature)
                                objects_list[cam][idx_obj].id = matched_obj_id

                    for cam in range(len(caps)):
                        for obj in objects_list[cam]:
                            if obj.id == -1:
                                obj.id = id_num
                                save_feature_to_db(obj.id, obj.feature)
                                unique_ids.add(id_num)
                                logging.debug(f"New object assigned ID {id_num}.")
                                id_num += 1

                    

                    for camera in range(len(caps)):
                        for obj in objects_list[camera]:
                            id = obj.id
                            color = generate_color_from_id(id)  # Use the new color generation function
                            xmin, ymin, xmax, ymax = obj.pos
                            cv2.rectangle(frames[camera], (xmin, ymin), (xmax, ymax), color, 2)
                            cv2.putText(frames[camera], str(id), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)

                    combined_frame = combine_frames(frames)
                    
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    unique_people_count = len(set(obj.id for cam in objects_list for obj in cam))
                    total_unique_people_count = get_unique_people_count()
                    cv2.putText(combined_frame, f"Unique People (Current Frame): {unique_people_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(combined_frame, f"Total Unique People: {total_unique_people_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if gui_supported:
                        cv2.imshow('Combined Frame', combined_frame)

        except cv2.error as e:
            logging.error(f"OpenCV Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        finally:
            release_video_captures(caps)
            if gui_supported:
                cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        logging.info("Starting the object tracking application.")
        sys.exit(main() or 0)
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        sys.exit(1)
