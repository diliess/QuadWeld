"""Preflight Cartesian IK, then execute smooth position-drive commands with feedback."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from planning.waypoint_generator import interpolate_pose
from robot.kinematics import pose_error


def distance_to_polyline(points, polyline):
    starts, ends = polyline[:-1], polyline[1:]
    edges = ends - starts
    squared_lengths = np.sum(edges * edges, axis=1)
    differences = np.asarray(points)[:, None, :] - starts
    fractions = np.clip(np.sum(differences * edges, axis=2) / np.maximum(squared_lengths, 1e-16), 0, 1)
    return np.linalg.norm(differences - fractions[:, :, None] * edges, axis=2).min(axis=1)


def plan_joint_path(model, path, initial_q, config, update_callback=None):
    initial_pose = model.forward(initial_q)
    approach = path.poses[0]
    count = max(2, int(np.ceil(np.linalg.norm(initial_pose[:3, 3] - approach[:3, 3]) / config.waypoint_spacing_m)),
                int(np.ceil(np.linalg.norm(pose_error(initial_pose, approach)[3:]) / 0.05)))
    transfer = [interpolate_pose(initial_pose, approach, t) for t in np.linspace(0, 1, count + 1)[1:]]
    poses = np.concatenate((transfer, path.poses[1:]))
    phases = ["approach"] * count + path.phases[1:]
    joints = [np.array(initial_q)]
    for i, pose in enumerate(poses):
        q = model.inverse(pose, joints[-1], position_tolerance=config.ik_position_tolerance_m,
                          orientation_tolerance=config.ik_orientation_tolerance_rad)
        if np.max(np.abs(q - joints[-1])) > config.max_joint_step_rad:
            raise RuntimeError(f"IK branch jump near waypoint {i}; change home pose or robot placement")
        # Check the complete tool and arm centreline against the plate slab and floor.
        # This is a conservative workspace check, not a general mesh/self-collision planner.
        for alpha in (0.0, 0.5, 1.0):
            sample_q = joints[-1] + alpha * (q - joints[-1])
            tcp, links = model.forward(sample_q, links=True)
            points = [link[:3, 3] for link in links[1:]] + [tcp[:3, 3]]
            for j, (a, b) in enumerate(zip(points, points[1:])):
                segment = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)) + 1))
                radius = 0.006 if j == len(points) - 2 else 0.035
                center, size = np.array(config.plate_center_world), np.array(config.plate_size)
                inside = np.all(np.abs(segment - center) <= size / 2 + radius, axis=1)
                if inside.any() or np.min(segment[:, 2]) < radius:
                    raise RuntimeError(f"Predicted arm/tool intersection with plate or floor at waypoint {i}")
        joints.append(q)
        if update_callback is not None:
            update_callback()
    return np.asarray(joints), np.concatenate(([initial_pose], poses)), ["home"] + phases


class RobotController:
    def __init__(self, articulation, model, simulation_app, config, output_dir):
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.simulation_manager import SimulationManager

        self.robot, self.model, self.app, self.config = articulation, model, simulation_app, config
        self.manager = SimulationManager
        self.output_dir = output_dir
        self.indices = [articulation.dof_names.index(name) for name in model.names]
        self.last_link = XformPrim(model.last_link_path)
        self.history = []
        self.trace_reference = None
        self.progress_callback = None
        self.robot.set_dof_gains(stiffnesses=config.drive_stiffness, dampings=config.drive_damping,
                                dof_indices=self.indices)
        self.robot.set_dof_max_velocities(config.joint_speed_rad_s, dof_indices=self.indices)

    def current_q(self):
        return self.robot.get_dof_positions(dof_indices=self.indices).numpy()[0].astype(float)

    def current_pose(self):
        position, quaternion = self.last_link.get_world_poses()
        pose = np.eye(4)
        pose[:3, 3] = position.numpy()[0]
        w, x, y, z = quaternion.numpy()[0]
        pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        return pose @ self.model.last_link_from_tcp

    def tick(self):
        if not self.app.is_running():
            raise RuntimeError("Simulation closed before trajectory completion")
        self.app.update()

    def initialize_home(self):
        q = np.array(self.config.home_joint_positions)
        # One reset-state initialization; all subsequent motion uses drive targets.
        self.robot.set_dof_positions(q, dof_indices=self.indices)
        self.robot.set_dof_velocities(np.zeros_like(q), dof_indices=self.indices)
        self.robot.set_dof_position_targets(q, dof_indices=self.indices)
        for _ in range(60):
            self.tick()
        error = pose_error(self.current_pose(), self.model.forward(self.current_q()))
        if np.linalg.norm(error[:3]) > 0.001 or np.linalg.norm(error[3:]) > 0.01:
            raise RuntimeError(f"USD FK / physics frame mismatch: {error.tolist()}")

    def move_to_pose(self, target_pose, previous_q, target_q, phase):
        cfg = self.config
        start_pose = self.model.forward(previous_q)
        distance = np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3])
        joint_distance = np.max(np.abs(target_q - previous_q))
        # Quintic interpolation stops at each sample/corner. Bounds include its peak derivatives.
        duration = max(0.08, 1.875 * distance / cfg.cartesian_speed_m_s,
                       1.875 * joint_distance / cfg.joint_speed_rad_s,
                       np.sqrt(5.8 * joint_distance / cfg.joint_acceleration_rad_s2))
        start = self.manager.get_simulation_time()
        elapsed, stable_steps, ticks = 0.0, 0, 0
        while elapsed < duration + cfg.waypoint_timeout_s:
            fraction = min(elapsed / duration, 1.0)
            blend = fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)
            command = previous_q + blend * (target_q - previous_q)
            self.robot.set_dof_position_targets(command, dof_indices=self.indices)
            self.tick()
            new_elapsed = self.manager.get_simulation_time() - start
            if new_elapsed <= elapsed:
                raise RuntimeError("Physics timeline stopped during execution")
            elapsed = new_elapsed
            actual = self.current_pose()
            reference = self.model.forward(command)
            error = pose_error(actual, reference)
            self.history.append([self.manager.get_simulation_time(), phase, *actual[:3, 3],
                                 *reference[:3, 3], np.linalg.norm(error[:3]), np.linalg.norm(error[3:])])
            if np.linalg.norm(error[:3]) > cfg.max_tracking_error_m:
                raise RuntimeError(f"Tracking diverged during {phase}: {np.linalg.norm(error[:3]):.4f} m")
            target_error = pose_error(actual, target_pose)
            reached = (elapsed >= duration and np.linalg.norm(target_error[:3]) <= cfg.position_tolerance_m
                       and np.linalg.norm(target_error[3:]) <= cfg.orientation_tolerance_rad)
            stable_steps = stable_steps + 1 if reached else 0
            if stable_steps >= 3:
                return
            ticks += 1
            if ticks % 120 == 0:
                print(f"  {phase}: error={np.linalg.norm(target_error[:3]):.4f} m", flush=True)
        raise RuntimeError(f"Timeout reaching {phase} waypoint {target_pose[:3, 3].tolist()}")

    def follow_cartesian_path(self, joints, poses, phases):
        success = False
        (self.output_dir / "execution_report.json").write_text(
            json.dumps({"completed": False, "status": "running"}, indent=2) + "\n")
        trace_indices = [i for i, phase in enumerate(phases) if phase == "trace"]
        if trace_indices:
            self.trace_reference = poses[trace_indices[0] - 1:trace_indices[-1] + 1, :3, 3]
        try:
            for i in range(1, len(joints)):
                if i == 1 or phases[i] != phases[i - 1] or i % 20 == 0:
                    print(f"Motion {i}/{len(joints) - 1}: {phases[i]}", flush=True)
                    if self.progress_callback:
                        self.progress_callback(f"Motion {i}/{len(joints) - 1}: {phases[i]}")
                self.move_to_pose(poses[i], joints[i - 1], joints[i], phases[i])
            success = True
        finally:
            # On failure hold the measured joint state; never continue to the next point.
            if not success:
                self.robot.set_dof_position_targets(self.current_q(), dof_indices=self.indices)
            self.save_report(success)

    def save_report(self, success):
        import csv

        with (self.output_dir / "tcp_execution.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "phase", "actual_x", "actual_y", "actual_z", "reference_x",
                             "reference_y", "reference_z", "position_error_m", "orientation_error_rad"])
            writer.writerows(self.history)
        errors = np.array([row[-2:] for row in self.history])
        report = {"completed": success, "status": "completed" if success else "failed", "samples": len(self.history)}
        if len(errors):
            report.update(max_position_error_m=float(errors[:, 0].max()),
                          rms_position_error_m=float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                          max_orientation_error_rad=float(errors[:, 1].max()))
        trace = np.array([row[2:5] for row in self.history if row[1] == "trace"])
        if len(trace) and self.trace_reference is not None:
            distances = distance_to_polyline(trace, self.trace_reference)
            clearance = trace[:, 2] - self.config.plate_top_z_world
            report.update(trace_max_path_error_m=float(distances.max()),
                          trace_rms_path_error_m=float(np.sqrt(np.mean(distances ** 2))),
                          trace_min_clearance_m=float(clearance.min()),
                          trace_max_clearance_m=float(clearance.max()))
        (self.output_dir / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
