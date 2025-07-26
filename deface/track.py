import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


class Track:
    """
    This class represents a single tracked object.
    """
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = Track.convert_bbox_to_z(bbox)
        self.last_hit = bbox
        self.time_since_update = 0
        self.id = track_id
        self.hits = 0
        self.hit_streak = 0
        self.best_hit_streak = 0
        self.age = 0
        self.last_hits_smoothing_buffer = [bbox]
        self.predictions_smoothing_buffer = []
        self.smoothing_window = 5
        self.fade_in = 5
        self.fade_out = 5

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top-left and x2,y2 is the bottom-right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

    @property
    def alpha(self):
        if self.time_since_update > 0 and self.fade_out > 0:
            return max(0, 1 - self.time_since_update / self.fade_out)
        elif self.fade_in > 0:
            return min(1, self.age / self.fade_in)
        return 1

    @property
    def last_hit_bbox(self):
        return np.concatenate((self.last_hit[:5], [self.alpha]), axis=None)

    @property
    def smoothed_hit_bbox(self):
        bbox = (np.mean(self.last_hits_smoothing_buffer, axis=0)
            if len(self.last_hits_smoothing_buffer) > 0
            else self.last_hit)
        return np.concatenate((bbox, [self.last_hit[4], self.alpha]), axis=None)

    @property
    def predicted_bbox(self):
        bbox = (np.mean(self.predictions_smoothing_buffer, axis=0)
            if len(self.predictions_smoothing_buffer) > 0
            else Track.convert_x_to_bbox(self.kf.x))
        return np.concatenate((bbox, [self.last_hit[4], self.alpha]), axis=None)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.best_hit_streak = max(self.best_hit_streak, self.hit_streak)

        self.last_hit = bbox
        self.kf.update(Track.convert_bbox_to_z(bbox))

        self.last_hits_smoothing_buffer.append(bbox)
        if len(self.last_hits_smoothing_buffer) > self.smoothing_window and self.smoothing_window > 1:
            self.last_hits_smoothing_buffer.pop(0)

    def predict(self):
        """
        Advances the state vector and use the new predicted position.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.last_hits_smoothing_buffer = []
        self.time_since_update += 1

        bbox = Track.convert_x_to_bbox(self.kf.x)
        self.predictions_smoothing_buffer.append(bbox)
        if len(self.predictions_smoothing_buffer) > self.smoothing_window and self.smoothing_window > 1:
            self.predictions_smoothing_buffer.pop(0)


class Tracker:
    """
    This class is the main tracker class.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, smoothing_window=5, fade_in=5, fade_out=5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.smoothing_window = smoothing_window
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0

    def update(self, dets):
        """
        This method is the main tracking method. It takes a list of detections and
        returns a list of bounding boxes for the tracked objects.
        """
        self.frame_count += 1
        # Predict locations from existing trackers.
        to_del = []
        ret = []
        for t, trk in enumerate(reversed(self.trackers)):
            trk.predict()
            if np.any(np.isnan(trk.predicted_bbox)):
                self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, self.trackers)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = Track(dets[i, :], self.next_id)
            self.next_id += 1
            trk.smoothing_window = self.smoothing_window
            trk.fade_in = self.fade_in
            trk.fade_out = self.fade_out
            self.trackers.append(trk)
        i = len(self.trackers)

        # select trackers to display
        for trk in reversed(self.trackers):
            if trk.time_since_update <= trk.fade_out and (trk.best_hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk)
            i -= 1

            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    def associate_detections_to_trackers(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = max(iou(det, trk.last_hit_bbox), iou(det, trk.predicted_bbox))

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if not matched_indices.size or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if not matched_indices.size or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
