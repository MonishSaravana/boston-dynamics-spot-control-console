"""Isolated SDK trajectory executor, disabled until supervised commissioning.

The caller owns an active lease, separate class E-stop, and a command worker.
This adapter cannot be reached from the default dry-run Spot console.
"""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .live_interaction import InteractionError, TimingBook

# Every trajectory command expires this soon after it is sent, so the robot
# stops on its own if the console or the network falls silent.
COMMAND_EXPIRY_S = .75
# How long a stale or failed pose read is tolerated before GO aborts. The
# command above has already expired by then, so the robot is standing still.
POSE_GRACE_S = 1.0
POSE_MAX_AGE_S = .6
# Spot's speed limit for GO, and the slowest average progress still waited for.
SPEED_LIMIT_MPS = .20
PROGRESS_FLOOR_MPS = .10
MIN_TIMEOUT_S = 15.
# Arrival judged from odometry alone when the SDK keeps reporting "near goal".
ARRIVED_XY_M = .20
ARRIVED_YAW_RAD = .25
ARRIVED_SETTLE_S = 1.0
# A route corner counts as passed this close; the next corner is commanded then.
WAYPOINT_PASSED_M = .30


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
        self.last_good_pose_s = None

    def _readiness(self, destination, walking=False):
        """Return the fresh pose, or None while a brief pose gap is tolerated."""
        try:
            pose = self.state_reader()
        except Exception as exc:
            pose = None
            reason = f"ROBOT STATE UNAVAILABLE: {exc}"
        else:
            reason = "ROBOT STATE STALE"
        now = self.clock()
        if pose is not None and not 0 <= now-pose.observed_host_s <= POSE_MAX_AGE_S:
            pose = None
        if pose is None:
            if (walking and self.last_good_pose_s is not None
                    and now-self.last_good_pose_s <= POSE_GRACE_S):
                return None
            raise InteractionError(reason)
        # The first dispatch needs Spot standing. Once our trajectory is active,
        # stepping, and the brief transition between the two, are expected states.
        upright = pose.standing or walking and (pose.stepping or pose.transitioning)
        if not upright or pose.world_frame != destination.world_frame:
            raise InteractionError("ROBOT STATE STALE or not standing")
        if not self.route_clear(destination):
            raise InteractionError("ROUTE BLOCKED or map stale")
        self.last_good_pose_s = now
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
        # Enough time for the route at half the speed limit plus turning and settling.
        timeout_s = max(MIN_TIMEOUT_S, 10.+distance_m/PROGRESS_FLOOR_MPS)
        start = self.clock()
        first_xy = None
        response_seen = False
        dispatched = False
        near_since = None
        self.last_good_pose_s = None
        from bosdyn.api import basic_command_pb2, geometry_pb2
        from bosdyn.client.robot_command import RobotCommandBuilder
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=SPEED_LIMIT_MPS, y=SPEED_LIMIT_MPS), angular=.30),
            min_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=-SPEED_LIMIT_MPS, y=-SPEED_LIMIT_MPS), angular=-.30)))
        # Spot walks each commanded point in a straight line, so a route that
        # bends is sent corner by corner; only the last corner carries arrival.
        waypoints = tuple(getattr(destination, "waypoints", ()) or
                          ((destination.x_m, destination.y_m, destination.yaw_rad),))
        commands = [RobotCommandBuilder.synchro_se2_trajectory_point_command(
            x, y, yaw, "odom", params=params) for x, y, yaw in waypoints]
        leg = 0
        processing = basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING
        try:
            while self.clock()-start < timeout_s:
                if self.cancel.is_set():
                    raise InteractionError("Stop requested")
                pose = self._readiness(destination, walking=dispatched)
                if pose is None:
                    # Pose gap inside the grace window: the last command expires on
                    # its own, so wait briefly for state instead of aborting.
                    if self.wait(.10):
                        raise InteractionError("Stop requested")
                    continue
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
                while (leg < len(waypoints)-1 and math.hypot(
                        pose.x_m-waypoints[leg][0], pose.y_m-waypoints[leg][1]) <= WAYPOINT_PASSED_M):
                    leg += 1
                command = commands[leg]
                final_leg = leg == len(waypoints)-1
                remaining_m = math.hypot(pose.x_m-destination.x_m, pose.y_m-destination.y_m)
                delta_yaw = math.atan2(math.sin(pose.yaw_rad-destination.yaw_rad),
                                       math.cos(pose.yaw_rad-destination.yaw_rad))
                at_goal_measured = remaining_m <= ARRIVED_XY_M and abs(delta_yaw) <= ARRIVED_YAW_RAD
                if at_goal_measured and dispatched:
                    near_since = near_since if near_since is not None else self.clock()
                    if self.clock()-near_since >= ARRIVED_SETTLE_S:
                        self.timings.add("measured_arrival", (self.clock()-start)*1000)
                        return "ARRIVED_MEASURED_POSE"
                else:
                    near_since = None
                dispatch_start = time.perf_counter()
                with self.command_lock:
                    if self.cancel.is_set():
                        raise InteractionError("Stop requested")
                    command_id = self.client.robot_command(
                        command=command, end_time_secs=time.time()+COMMAND_EXPIRY_S, timeout=.7)
                self.timings.add("command_rpc_ack", (time.perf_counter()-dispatch_start)*1000)
                self.command_ids.append(command_id)
                dispatched = True
                feedback_start = time.perf_counter()
                feedback = self.client.robot_command_feedback(command_id, timeout=.7)
                self.timings.add("feedback_rpc", (time.perf_counter()-feedback_start)*1000)
                mobility = feedback.feedback.synchronized_feedback.mobility_command_feedback
                trajectory = mobility.se2_trajectory_feedback
                near_goal = getattr(trajectory, "STATUS_NEAR_GOAL", None)
                reached = trajectory.status == trajectory.STATUS_AT_GOAL or (
                    near_goal is not None and trajectory.status == near_goal)
                if (final_leg and mobility.status == processing and reached
                        and trajectory.body_movement_status == trajectory.BODY_STATUS_SETTLED):
                    if remaining_m > .3 or abs(delta_yaw) > .5:
                        raise InteractionError("ROBOT RESPONSE MISMATCH: feedback/odom disagree")
                    self.timings.add("measured_arrival", (self.clock()-start)*1000)
                    return "ARRIVED_MEASURED_FEEDBACK"
                if mobility.status != processing:
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
            # The previous trajectory expires within COMMAND_EXPIRY_S if the network is gone.
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
