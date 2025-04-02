from typing import Any, Dict, List

import cv2
import numpy as np

from ...object_tracker import ObjectTracker
from . import matching
from .dtypes import STrack, TrackState
from .kalman_filter import KalmanFilter
from .utils import joint_stracks, remove_duplicate_stracks, sub_stracks


class BYTETracker(ObjectTracker):
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        mot20: bool = False,
        frame_rate: int = 30,
        aspect_ratio_thresh: float = 1.6,
        min_box_area: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0

        # self.det_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.results = []

        self.mot20 = mot20
        self.track_buffer = track_buffer
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray):
        bboxes = np.array([detection["bbox"] for detection in detections])
        scores = np.array([detection["confidence"] for detection in detections])
        class_ids = np.array([detection["class_id"] for detection in detections])
        detection_ids = np.array([detection["detection_id"] for detection in detections])

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # if output_results.shape[1] == 5:
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]
        # else:
        #     output_results = output_results.cpu().numpy()
        #     scores = output_results[:, 4] * output_results[:, 5]
        #     bboxes = output_results[:, :4]  # x1y1x2y2

        # if img_info and img_size:
        #     img_h, img_w = img_info[0], img_info[1]
        #     scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        #     bboxes = bboxes / scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]

        det_ids = detection_ids[remain_inds]
        det_ids_second = detection_ids[inds_second]

        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    tlwh=STrack.tlbr_to_tlwh(tlbr),
                    det_id=det_id,
                    score=s,
                    class_id=class_id,
                )
                for (tlbr, det_id, s, class_id) in zip(dets, det_ids, scores_keep, class_ids_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(tlwh=STrack.tlbr_to_tlwh(tlbr), det_id=det_id, score=s, class_id=class_id)
                for (tlbr, det_id, s, class_id) in zip(dets_second, det_ids_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.update_crops(frame)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # visualize tracks
        if self.visualize:
            self.visualize_tracks(frame)

        return self._get_track_messages()

    def _get_track_messages(self) -> List[Dict[str, Any]]:
        """Gets a list of track messages.

        Returns:
            A list of track messages.
        """
        tracked_strack_messages = [t.get_track_message() for t in self.tracked_stracks]
        removed_strack_messages = [t.get_track_message() for t in self.removed_stracks]
        lost_strack_messages = [t.get_track_message() for t in self.lost_stracks]
        track_messages = tracked_strack_messages + removed_strack_messages + lost_strack_messages
        return track_messages

    def visualize_tracks(self, frame: np.ndarray, thickness: int = 2):
        online_targets = [track for track in self.tracked_stracks if track.is_activated]
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            cid = t.class_id
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                tx1, ty1, tw, th = tlwh.astype(int)
                track_str = f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"

                # reshape bounding box to image
                x1 = max(0, tx1)
                y1 = max(0, ty1)
                x2 = min(frame.shape[1], tx1 + tw)
                y2 = min(frame.shape[0], ty1 + th)

                cv2.putText(
                    frame,
                    f"{self.names[cid]} {str(tid)}",
                    (tx1, ty1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.class_colors[cid],
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_colors[cid], thickness)

                det = frame[y1:y2, x1:x2, :].copy()
                det_mask = np.ones(det.shape, dtype=np.uint8) * np.uint8(self.class_colors[cid])
                res = cv2.addWeighted(det, 0.6, det_mask, 0.4, 1.0)
                frame[y1:y2, x1:x2] = res
                self.results.append(track_str)
