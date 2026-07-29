import sys
import types

# Stub PyNomad so importing the optimizer does not require the real package.
py_nomad = types.ModuleType("PyNomad")
py_nomad.optimize = lambda *args, **kwargs: {"status": "ok"}
sys.modules.setdefault("PyNomad", py_nomad)

# Stub TrackEval internals for import-time safety.
trackeval = types.ModuleType("trackeval")


class DummyDataset:
    @staticmethod
    def get_default_dataset_config():
        return {}


class DummyMetric:
    def __init__(self, *args, **kwargs):
        pass


class DummyEvaluator:
    @staticmethod
    def get_default_eval_config():
        return {}


trackeval.datasets = types.SimpleNamespace(MotChallenge2DBox=DummyDataset)
trackeval.metrics = types.SimpleNamespace(HOTA=DummyMetric, CLEAR=DummyMetric, Identity=DummyMetric)
trackeval.Evaluator = DummyEvaluator
sys.modules.setdefault("trackeval", trackeval)

# Minimal stubs for Ultralytics and OpenCV so imports succeed in tests.
ultralytics_mod = types.ModuleType("ultralytics")


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        self.names = {0: "person"}


ultralytics_mod.YOLO = DummyYOLO
sys.modules.setdefault("ultralytics", ultralytics_mod)

ultralytics_utils = types.ModuleType("ultralytics.utils")
ultralytics_plotting = types.ModuleType("ultralytics.utils.plotting")
ultralytics_plotting.colors = lambda cls, _unused: (255, 0, 0)
ultralytics_utils.plotting = ultralytics_plotting
sys.modules.setdefault("ultralytics.utils", ultralytics_utils)
sys.modules.setdefault("ultralytics.utils.plotting", ultralytics_plotting)

cv2_mod = types.ModuleType("cv2")
cv2_mod.VideoCapture = object
cv2_mod.CAP_PROP_FRAME_WIDTH = 0
cv2_mod.CAP_PROP_FRAME_HEIGHT = 0
cv2_mod.CAP_PROP_FRAME_COUNT = 0
cv2_mod.CAP_PROP_FPS = 0
cv2_mod.CAP_PROP_POS_FRAMES = 0
cv2_mod.FONT_HERSHEY_SIMPLEX = 0
cv2_mod.LINE_AA = 0
cv2_mod.rectangle = lambda *args, **kwargs: None
cv2_mod.getTextSize = lambda *args, **kwargs: ((0, 0), None)
cv2_mod.putText = lambda *args, **kwargs: None
cv2_mod.addWeighted = lambda *args, **kwargs: None
cv2_mod.destroyAllWindows = lambda: None
cv2_mod.VideoWriter = object
cv2_mod.VideoWriter_fourcc = lambda *args, **kwargs: None
cv2_mod.imshow = lambda *args, **kwargs: None
cv2_mod.waitKey = lambda *args, **kwargs: 0
sys.modules.setdefault("cv2", cv2_mod)