"""Isolated SDK trajectory executor, disabled until supervised commissioning.

The caller owns an active lease, separate class E-stop, and a command worker.
This adapter cannot be reached from the default dry-run Spot console.
"""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .live_interaction import InteractionError, TimingBook


class SpotTrajectoryExecutor:
    mode = "SPOT_SUPERVISED"

    def __init__(self, command_client, state_reader, route_clear, *, enabled=False,
                 clock=time.monotonic, wait=None):
        self.client = command_client
        self.state_reader = state_reader
        self.route_clear = route_clear
        self.enabled = enabled
        self.clock = clock
        self.cancel = threading.Event()
        self.command_lock = threading.RLock()
        self.wait = wait or self.cancel.wait
        self.command_ids = []
        self.measured = []
        self.timings = TimingBook()

    def _readiness(self, destination):
        pose = self.state_reader()
        if (pose is None or not pose.standing or pose.world_frame != destination.world_frame
                or not 0 <= self.clock()-pose.observed_host_s <= .6):
            raise InteractionError("ROBOT STATE STALE or not standing")
        if not self.route_clear(destination):
            raise InteractionError("ROUTE BLOCKED or map stale")
        return pose

    def execute(self, destination):
        if not self.enabled:
            raise InteractionError("NAVIGATION UNAVAILABLE: supervised Spot GO is disabled")
        if destination.world_frame != "odom":
            raise InteractionError("NAVIGATION UNAVAILABLE: only odom-frame goals supported")
        if not destination.route_xy_m:
            raise InteractionError("NAVIGATION UNAVAILABLE: no previewed route")
        distance_m = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in
                         zip(destination.route_xy_m, destination.route_xy_m[1:]))
        if distance_m > 2.5:
            raise InteractionError("NAVIGATION UNAVAILABLE: first-run route exceeds 2.5 m")
        start = self.clock()
        first_xy = None
        response_seen = False
        from bosdyn.api import basic_command_pb2, geometry_pb2
        from bosdyn.client.robot_command import RobotCommandBuilder
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=.20, y=.20),
                                               angular=.30),
            min_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=-.20, y=-.20),
                                               angular=-.30)))
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            destination.x_m, destination.y_m, destination.yaw_rad, "odom", params=params)
        try:
            while self.clock()-start < 15.:
                if self.cancel.is_set():
                    raise InteractionError("Stop requested")
                pose = self._readiness(destination)
                if min(math.hypot(pose.x_m-x, pose.y_m-y)
                       for x, y in destination.route_xy_m) > .6:
                    raise InteractionError("ROBOT DEVIATED FROM PREVIEW")
                self.measured.append((self.clock(), pose.x_m, pose.y_m, pose.yaw_rad))
                if len(self.measured) > 1000:
                    del self.measured[:-1000]
                if first_xy is None:
                    first_xy = (pose.x_m, pose.y_m)
                elif not response_seen and math.hypot(pose.x_m-first_xy[0], pose.y_m-first_xy[1]) > .02:
                    self.timings.add("measured_robot_response", (self.clock()-start)*1000)
                    response_seen = True
                dispatch_start = time.perf_counter()
                with self.command_lock:
                    if self.cancel.is_set():
                        raise InteractionError("Stop requested")
                    command_id = self.client.robot_command(
                        command=command, end_time_secs=time.time()+.75, timeout=.7)
                self.timings.add("command_rpc_ack", (time.perf_counter()-dispatch_start)*1000)
                self.command_ids.append(command_id)
                feedback_start = time.perf_counter()
                feedback = self.client.robot_command_feedback(command_id, timeout=.7)
                self.timings.add("feedback_rpc", (time.perf_counter()-feedback_start)*1000)
                mobility = feedback.feedback.synchronized_feedback.mobility_command_feedback
                trajectory = mobility.se2_trajectory_feedback
                if (mobility.status == basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING
                        and trajectory.status == trajectory.STATUS_AT_GOAL
                        and trajectory.body_movement_status == trajectory.BODY_STATUS_SETTLED):
                    delta_yaw = math.atan2(math.sin(pose.yaw_rad-destination.yaw_rad),
                                           math.cos(pose.yaw_rad-destination.yaw_rad))
                    if (math.hypot(pose.x_m-destination.x_m, pose.y_m-destination.y_m) > .3 or
                            abs(delta_yaw) > .5):
                        raise InteractionError("ROBOT RESPONSE MISMATCH: feedback/odom disagree")
                    self.timings.add("measured_arrival", (self.clock()-start)*1000)
                    return "ARRIVED_MEASURED_FEEDBACK"
                if mobility.status != basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING:
                    raise InteractionError("NAVIGATION FAILED: SDK mobility feedback")
                if self.wait(.20):
                    raise InteractionError("Stop requested")
            raise InteractionError("NAVIGATION TIMEOUT")
        finally:
            self._zero()

    def _zero(self):
        from bosdyn.client.robot_command import RobotCommandBuilder
        try:
            self.client.robot_command(
                command=RobotCommandBuilder.synchro_velocity_command(0, 0, 0),
                end_time_secs=time.time()+.35, timeout=.7)
        except Exception:
            # The previous trajectory expires within .75 s if the network is gone.
            pass

    def stop(self):
        self.cancel.set()
        with self.command_lock:
            self._zero()


class AsyncSpotExecutor:
    """Keep the UI and its Stop request responsive during a trajectory."""

    mode = "SPOT_SUPERVISED"

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scope-navigation")
        self.future = None

    def execute(self, destination):
        if self.future is not None and not self.future.done():
            raise InteractionError("Navigation already active")
        with self.trajectory.command_lock:
            self.trajectory.cancel.clear()
        self.future = self.pool.submit(self.trajectory.execute, destination)
        return "NAVIGATION_STARTED; measured feedback pending"

    def status(self):
        if self.future is None:
            return "IDLE"
        if not self.future.done():
            return "MOVING; measured feedback pending"
        try:
            return self.future.result()
        except Exception as exc:
            return f"FAILED: {exc}"

    def stop(self):
        self.trajectory.stop()

    def close(self):
        self.stop()
        self.pool.shutdown(wait=True, cancel_futures=True)
