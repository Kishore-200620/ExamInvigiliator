import numpy as np
from filterpy.kalman import KalmanFilter

from app.ai.detector.models import BoundingBox
from app.ai.tracker.base import MotionModel
from app.ai.tracker.motion import MotionState


class KalmanMotionModel(MotionModel):
    """
    Constant Velocity Kalman Motion Model.

    State Vector

    x
    y
    w
    h
    vx
    vy
    vw
    vh
    """

    def __init__(self):

        self.filter = KalmanFilter(
            dim_x=8,
            dim_z=4,
        )

        self._configure()

        self.motion_state: MotionState | None = None

    def _configure(self):

        dt = 1.0

        self.filter.F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],

            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ])

        self.filter.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ])

        self.filter.P *= 10

        self.filter.R *= 1

        self.filter.Q *= 0.01

    def initialize(
        self,
        bbox: BoundingBox,
    ) -> None:

        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2

        w = bbox.width
        h = bbox.height

        self.filter.x = np.array([
            cx,
            cy,
            w,
            h,
            0,
            0,
            0,
            0,
        ]).reshape(8, 1)

    def predict(self) -> BoundingBox:

        self.filter.predict()

        state = self.filter.x.flatten()

        cx = state[0]
        cy = state[1]
        w = state[2]
        h = state[3]

        bbox = BoundingBox(
            x1=int(cx - w / 2),
            y1=int(cy - h / 2),
            x2=int(cx + w / 2),
            y2=int(cy + h / 2),
        )

        return bbox

    def update(
        self,
        bbox: BoundingBox,
    ) -> None:

        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2

        w = bbox.width
        h = bbox.height

        measurement = np.array([
            cx,
            cy,
            w,
            h,
        ])

        self.filter.update(measurement)