import os
import time


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_bool(name, default):
    value = str(os.environ.get(name, str(default))).lower().strip()
    return value in ("1", "true", "yes", "on")


class TrafficLightPipeline:
    """
    Teknofest/CARLA trafik ışığı algısı için tek geçişli pipeline.

    Kritik prensip:
      - Detection state sadece burada belirlenir.
      - Draw/debug/active seçimi detection'ı tekrar mutate etmez.
      - Decision tarafına sadece aktif trafik ışığı gönderilir.
    """

    KNOWN = {"red", "yellow", "green"}

    def __init__(self, node):
        self.node = node
        self.last_active = None
        self.last_active_time = 0.0

    def state_conf(self, det):
        for key in (
            "traffic_light_state_confidence",
            "tl_state_confidence",
            "state_confidence",
            "state_conf",
            "classifier_confidence",
        ):
            value = det.get(key, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

        probs = det.get("traffic_light_state_probs", None)
        state = str(det.get("traffic_light_state", "unknown")).lower().strip()
        if isinstance(probs, dict) and state in probs:
            try:
                return float(probs[state])
            except Exception:
                pass

        return 0.0

    def best_classifier_prob(self, probs):
        if not isinstance(probs, dict):
            return "unknown", 0.0

        best_state = "unknown"
        best_conf = 0.0
        for state in ("red", "yellow", "green"):
            try:
                conf = float(probs.get(state, 0.0))
            except Exception:
                conf = 0.0
            if conf > best_conf:
                best_state = state
                best_conf = conf
        return best_state, best_conf

    def bbox_geom(self, det, frame_w, frame_h):
        x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area = bw * bh
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return {
            "bw": bw,
            "bh": bh,
            "area": area,
            "cx": cx,
            "cy": cy,
            "cx_ratio": cx / max(1.0, float(frame_w)),
            "cy_ratio": cy / max(1.0, float(frame_h)),
        }

    def in_active_roi(self, geom):
        if not _env_bool("TL_ACTIVE_ROI_ENABLED", True):
            return True

        x_min = _env_float("TL_ACTIVE_ROI_X_MIN", 0.30)
        x_max = _env_float("TL_ACTIVE_ROI_X_MAX", 0.88)
        y_min = _env_float("TL_ACTIVE_ROI_Y_MIN", 0.04)
        y_max = _env_float("TL_ACTIVE_ROI_Y_MAX", 0.60)

        return (
            x_min <= geom["cx_ratio"] <= x_max
            and y_min <= geom["cy_ratio"] <= y_max
        )

    def accept_state_from_probs_if_needed(self, result):
        state = str(result.get("state", "unknown")).lower().strip()
        probs = result.get("classifier_probs", {}) or {}
        best_state, best_conf = self.best_classifier_prob(probs)

        if state in self.KNOWN:
            conf = result.get("state_confidence")
            if conf is None:
                conf = best_conf
            return state, float(conf or 0.0), "classifier_final"

        if best_state == "red" and best_conf >= _env_float("TL_DIRECT_KEEP_RED_CONF", 0.55):
            return "red", best_conf, "classifier_probs_red_keep"

        if best_state == "green" and best_conf >= _env_float("TL_DIRECT_KEEP_GREEN_CONF", 0.70):
            return "green", best_conf, "classifier_probs_green_keep"

        if best_state == "yellow" and best_conf >= _env_float("TL_DIRECT_KEEP_YELLOW_CONF", 0.90):
            return "yellow", best_conf, "classifier_probs_yellow_keep"

        return "unknown", 0.0, "unknown"

    def enrich_candidate(self, frame, det):
        out = dict(det)

        try:
            result = self.node.classify_traffic_light_state(
                frame,
                out.get("bbox", [0, 0, 0, 0]),
            )
        except Exception as exc:
            result = {
                "state": "unknown",
                "source": "tl_pipeline_classifier_error",
                "state_confidence": 0.0,
                "classifier_probs": {},
                "classifier_reason": f"error:{exc}",
                "hsv_state": "unknown",
                "hsv_scores": {},
                "hsv_reason": "classifier_error",
                "final_reason": f"tl_pipeline_classifier_error:{exc}",
            }

        state, state_conf, keep_source = self.accept_state_from_probs_if_needed(result)

        out["traffic_light_state"] = state
        out["traffic_light_state_confidence"] = float(state_conf)
        out["state_confidence"] = float(state_conf)
        out["state_conf"] = float(state_conf)

        base_source = str(result.get("source", "unknown"))
        if keep_source != "classifier_final":
            base_source = keep_source

        out["traffic_light_state_source"] = base_source
        out["state_source"] = base_source
        out["traffic_light_state_probs"] = result.get("classifier_probs", {}) or {}
        out["traffic_light_state_classifier_reason"] = result.get("classifier_reason", "-")
        out["traffic_light_hsv_state"] = result.get("hsv_state", "unknown")
        out["traffic_light_color_source"] = base_source
        out["traffic_light_color_scores"] = result.get("hsv_scores", {}) or {}
        out["traffic_light_color_reason"] = result.get("final_reason", "-")
        out["traffic_light_hsv_reason"] = result.get("hsv_reason", "-")

        return out

    def is_candidate_usable(self, det, frame_w, frame_h):
        state = str(det.get("traffic_light_state", "unknown")).lower().strip()
        if state not in self.KNOWN:
            det["tl_pipeline_reject_reason"] = "unknown_state"
            return False

        conf = self.state_conf(det)
        det_conf = float(det.get("confidence", 0.0))
        geom = self.bbox_geom(det, frame_w, frame_h)

        min_w = _env_float("TL_PIPELINE_MIN_W", 4.0)
        min_h = _env_float("TL_PIPELINE_MIN_H", 4.0)
        min_area = _env_float("TL_PIPELINE_MIN_AREA", 16.0)

        if geom["bw"] < min_w or geom["bh"] < min_h or geom["area"] < min_area:
            if state == "green" and conf >= _env_float("TL_DIRECT_KEEP_GREEN_CONF", 0.70):
                return True
            if state == "red" and conf >= _env_float("TL_DIRECT_KEEP_RED_CONF", 0.55):
                return True
            det["tl_pipeline_reject_reason"] = (
                f"too_small:bw={geom['bw']:.1f},bh={geom['bh']:.1f},area={geom['area']:.1f}"
            )
            return False

        if det_conf < _env_float("TL_PIPELINE_MIN_DET_CONF", 0.05):
            det["tl_pipeline_reject_reason"] = f"low_det_conf:{det_conf:.3f}"
            return False

        min_state_conf = {
            "red": _env_float("TL_PIPELINE_RED_MIN_STATE_CONF", 0.45),
            "yellow": _env_float("TL_PIPELINE_YELLOW_MIN_STATE_CONF", 0.65),
            "green": _env_float("TL_PIPELINE_GREEN_MIN_STATE_CONF", 0.55),
        }.get(state, 0.50)

        if conf < min_state_conf:
            det["tl_pipeline_reject_reason"] = (
                f"low_state_conf:{state}:{conf:.3f}<={min_state_conf:.3f}"
            )
            return False

        return True

    def score_candidate(self, det, frame_w, frame_h):
        geom = self.bbox_geom(det, frame_w, frame_h)
        state = str(det.get("traffic_light_state", "unknown")).lower().strip()
        state_conf = self.state_conf(det)
        det_conf = float(det.get("confidence", 0.0))
        in_roi = self.in_active_roi(geom)

        target_x = _env_float("TL_ACTIVE_TARGET_X_RATIO", 0.52)
        target_y = _env_float("TL_ACTIVE_TARGET_Y_RATIO", 0.22)

        x_score = 1.0 - min(1.0, abs(geom["cx_ratio"] - target_x) / 0.50)
        y_score = 1.0 - min(1.0, abs(geom["cy_ratio"] - target_y) / 0.45)
        size_score = min(1.0, geom["area"] / _env_float("TL_ACTIVE_SIZE_NORM_AREA", 450.0))

        # Renk önceliği nötr. Doğru yeşili yan/uzak kırmızı ezmesin.
        state_priority = 1.0 if state in self.KNOWN else 0.0

        return (
            1 if in_roi else 0,
            round(x_score, 4),
            round(y_score, 4),
            round(size_score, 4),
            round(state_conf, 4),
            round(det_conf, 4),
            state_priority,
        )

    def select_active(self, candidates, frame_w, frame_h):
        usable = []
        rejected = []

        for det in candidates:
            geom = self.bbox_geom(det, frame_w, frame_h)
            det["tl_active_roi"] = bool(self.in_active_roi(geom))
            det["tl_cx_ratio"] = round(geom["cx_ratio"], 4)
            det["tl_cy_ratio"] = round(geom["cy_ratio"], 4)

            if self.is_candidate_usable(det, frame_w, frame_h):
                usable.append(det)
            else:
                rejected.append(det)

        if not usable:
            return None, rejected

        roi_usable = [d for d in usable if d.get("tl_active_roi", False)]
        pool = roi_usable if roi_usable else usable

        scored = []
        for det in pool:
            score = self.score_candidate(det, frame_w, frame_h)
            det["tl_active_score"] = score
            scored.append((score, det))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], rejected

    def process(self, frame, detections, frame_w, frame_h):
        out = []
        tl_candidates = []

        for det in detections:
            if det.get("label") != "traffic_light":
                out.append(det)
                continue

            enriched = self.enrich_candidate(frame, det)
            tl_candidates.append(enriched)

        active, rejected = self.select_active(tl_candidates, frame_w, frame_h)

        for det in tl_candidates:
            det["active_traffic_light"] = False

        info = {
            "state": "unknown",
            "confidence": None,
            "state_confidence": None,
            "state_source": None,
            "state_probs": None,
            "color_scores": None,
            "color_reason": None,
            "bbox": None,
            "candidate_count": len(tl_candidates),
            "rejected_count": len(rejected),
        }

        if active is not None:
            active["active_traffic_light"] = True

            info = {
                "state": active.get("traffic_light_state", "unknown"),
                "confidence": float(active.get("confidence", 0.0)),
                "state_confidence": self.state_conf(active),
                "state_source": active.get("traffic_light_state_source", active.get("state_source", None)),
                "state_probs": active.get("traffic_light_state_probs", None),
                "color_scores": active.get("traffic_light_color_scores", None),
                "color_reason": active.get("traffic_light_color_reason", None),
                "bbox": active.get("bbox", None),
                "candidate_count": len(tl_candidates),
                "rejected_count": len(rejected),
                "score": active.get("tl_active_score", None),
                "roi": active.get("tl_active_roi", None),
            }

            self.last_active = dict(active)
            self.last_active_time = time.time()

        if _env_bool("TL_KEEP_ONLY_ACTIVE", True):
            if active is not None:
                out.append(active)
        else:
            out.extend(tl_candidates)

        return out, info, tl_candidates, rejected
