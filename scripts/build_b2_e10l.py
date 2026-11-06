"""Compose Unitree B2 and the E10L-Pro welding arm into one URDF, then convert it to USD.

    python scripts/build_b2_e10l.py              # the USD step needs Isaac Sim 6 and a GPU
    python scripts/build_b2_e10l.py --urdf-only

Sources stay under assets/vendor/ (see assets/robots/b2_e10l/README.md). The composed URDF is
written to assets/vendor/b2_e10l/, the USD and manifest.json to assets/robots/b2_e10l/.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
B2_DIR = ROOT / "assets/vendor/unitree_ros/robots/b2_description"
E10L_DIR = ROOT / "assets/vendor/e10l/final_urdf"
E10L_ARCHIVE = ROOT / "assets/vendor/e10l/带焊接枪E10-L机械臂模型.zip"
WORK_DIR = ROOT / "assets/vendor/b2_e10l"
OUTPUT_DIR = ROOT / "assets/robots/b2_e10l"
NAME = "B2-E10L"
PREFIX = "e10l_"

# Assumed aluminium adapter on the trunk centre, clear of the cover (visual top z=0.088 m).
PLATE_SIZE = (0.26, 0.24, 0.02)
PLATE_BOTTOM = (0.0, 0.0, 0.09)
ARM_MASS_KG = 45.0  # Huayan E10L-Pro datasheet; the exported CAD inertials sum to 29.4 kg.
TORCH_MASS_KG = 1.5  # Assumed: the supplied torch model has no mass.
JOINT_SPEED_DEG_S = (100, 100, 150, 150, 180, 180)  # Datasheet, J1-J6.
EFFORT_MARGIN = 1.5  # Over the worst static gravity torque; no torque data was supplied.
# Gravity gives J1 nothing; the floor is a published 10 kg-class cobot rating (UR10e), not E10L data.
EFFORT_FLOOR_NM = (330, 330, 150, 56, 56, 56)
GRAVITY = np.array([0., 0., -9.81])


def origin_matrix(node):
    result = np.eye(4)
    if node is not None:
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    return result


def origin_element(matrix, tag="origin"):
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    return ET.Element(tag, xyz=" ".join(f"{v:.9g}" for v in matrix[:3, 3]), rpy=" ".join(f"{v:.9g}" for v in rpy))


def stl_vertices(path):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * count != len(data):
        raise ValueError(f"Expected a binary STL: {path}")
    records = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=count)
    return records["v"].reshape(-1, 3, 3).astype(float)


def torch_frames():
    """TCP at the wire tip, +Z out of the straight nozzle; shell inertia scaled to the assumed mass."""
    triangles = stl_vertices(E10L_DIR / "meshes/welding_torch.STL")
    vertices = triangles.reshape(-1, 3)
    # The nozzle end is the highest region in the flange frame; refine the tip as the extreme
    # vertex along the nozzle's own axis, since the nozzle is inclined to the flange.
    tip = vertices[vertices[:, 2].argmax()]
    for _ in range(10):
        nozzle = vertices[np.linalg.norm(vertices - tip, axis=1) < 0.05]
        centre = nozzle.mean(0)
        axis = np.linalg.eigh(np.cov((nozzle - centre).T))[1][:, -1]
        axis *= np.sign(axis @ (tip - centre))
        near = vertices[np.linalg.norm(vertices - tip, axis=1) < 0.08]
        refined = near[(near @ axis).argmax()]
        if np.allclose(refined, tip):
            break
        tip = refined
    else:
        raise RuntimeError("Torch tip did not converge")
    side = np.cross(axis, [0., 0., 1.])
    side /= np.linalg.norm(side)
    tcp = np.eye(4)
    tcp[:3, :3] = np.column_stack((np.cross(side, axis), side, axis))
    tcp[:3, 3] = tip
    centroids = triangles.mean(1)
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    weights = TORCH_MASS_KG * areas / areas.sum()
    com = weights @ centroids / TORCH_MASS_KG
    r = centroids - com
    inertia = np.einsum("i,ij,ik->jk", weights, r, r)
    inertia = np.trace(inertia) * np.eye(3) - inertia
    return tcp, com, inertia


def inertial_element(mass, com, inertia):
    node = ET.Element("inertial")
    node.append(ET.Element("origin", xyz=" ".join(f"{v:.9g}" for v in com), rpy="0 0 0"))
    node.append(ET.Element("mass", value=f"{mass:.9g}"))
    node.append(ET.Element("inertia", **{key: f"{inertia[i, j]:.9g}" for key, (i, j) in
                                        {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2),
                                         "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}.items()}))
    return node


def relative_mesh(source_dir, filename):
    path = (Path(source_dir) / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return os.path.relpath(path, WORK_DIR)


def b2_elements():
    """B2 as published, without its massless sensor frames; the lidar collider joins the trunk."""
    source = ET.parse(B2_DIR / "urdf/b2_description.urdf").getroot()
    links = {link.get("name"): link for link in source.findall("link")}
    joints = source.findall("joint")
    massless = {name for name, link in links.items() if float(link.find("inertial/mass").get("value")) < 1e-6}
    trunk = copy.deepcopy(links["base_link"])
    for joint in joints:
        child = joint.find("child").get("link")
        if child in massless:
            if joint.find("parent").get("link") != "base_link" or joint.get("type") != "fixed":
                raise ValueError(f"Unexpected massless link {child}")
            for collision in links[child].findall("collision"):
                moved = copy.deepcopy(collision)
                matrix = origin_matrix(joint.find("origin")) @ origin_matrix(collision.find("origin"))
                for node in moved.findall("origin"):
                    moved.remove(node)
                moved.insert(0, origin_element(matrix))
                trunk.append(moved)
    result = [trunk] + [copy.deepcopy(link) for name, link in links.items() if name not in massless | {"base_link"}]
    result += [copy.deepcopy(joint) for joint in joints if joint.find("child").get("link") not in massless]
    for mesh in (node for element in result for node in element.iter("mesh")):
        mesh.set("filename", relative_mesh(B2_DIR, mesh.get("filename").removeprefix("package://b2_description/")))
    return result, massless


def arm_elements(torch_com, torch_inertia):
    source = ET.parse(E10L_DIR / "E10L_pro_with_welding_torch.urdf").getroot()
    cad_mass = sum(float(link.find("inertial/mass").get("value")) for link in source.findall("link")
                   if link.find("inertial") is not None)
    scale = ARM_MASS_KG / cad_mass
    links, joints = [], []
    for link in source.findall("link"):
        link = copy.deepcopy(link)
        name = link.get("name")
        link.set("name", PREFIX + name)
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            mass.set("value", f"{float(mass.get('value')) * scale:.9g}")
            inertia = inertial.find("inertia")
            for key, value in inertia.attrib.items():
                inertia.set(key, f"{float(value) * scale:.9g}")
        else:
            link.insert(0, inertial_element(TORCH_MASS_KG, torch_com, torch_inertia))
        for index, material in enumerate(link.iter("material")):
            material.set("name", f"{PREFIX}{name}_{index}")
        for mesh in link.iter("mesh"):
            mesh.set("filename", relative_mesh(E10L_DIR, mesh.get("filename")))
        links.append(link)
    revolute = 0
    for joint in source.findall("joint"):
        joint = copy.deepcopy(joint)
        joint.set("name", PREFIX + joint.get("name"))
        for tag in ("parent", "child"):
            joint.find(tag).set("link", PREFIX + joint.find(tag).get("link"))
        if joint.get("type") == "revolute":
            joint.find("limit").set("velocity", f"{math.radians(JOINT_SPEED_DEG_S[revolute]):.9g}")
            revolute += 1
        joints.append(joint)
    if revolute != 6:
        raise ValueError(f"Expected six revolute arm joints, found {revolute}")
    efforts = gravity_efforts(links, joints)
    for joint, effort in zip((j for j in joints if j.get("type") == "revolute"), efforts):
        joint.find("limit").set("effort", f"{effort:g}")
    return links, joints, cad_mass, efforts


def gravity_efforts(links, joints, samples=4000, seed=7):
    """Worst static gravity torque over random poses, with margin, rounded up to 10 N m."""
    inertials = {link.get("name"): (float(link.find("inertial/mass").get("value")),
                                    np.fromstring(link.find("inertial/origin").get("xyz"), sep=" "))
                 for link in links}
    chain = [(joint, origin_matrix(joint.find("origin"))) for joint in joints]
    revolute = [joint for joint, _ in chain if joint.get("type") == "revolute"]
    limits = np.array([[float(j.find("limit").get(k)) for k in ("lower", "upper")] for j in revolute])
    worst = np.zeros(len(revolute))
    rng = np.random.default_rng(seed)
    for q in rng.uniform(limits[:, 0], limits[:, 1], size=(samples, len(revolute))):
        frames = {PREFIX + "base_link": np.eye(4)}
        axes, index = [], 0
        for joint, fixed in chain:
            parent = frames[joint.find("parent").get("link")] @ fixed
            motion = np.eye(4)
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                axes.append((parent[:3, 3], parent[:3, :3] @ axis, joint.find("child").get("link")))
                index += 1
            frames[joint.find("child").get("link")] = parent @ motion
        order = [joint.find("child").get("link") for joint, _ in chain]
        for j, (point, axis, child) in enumerate(axes):
            torque = 0.
            for name in order[order.index(child):]:
                mass, com = inertials[name]
                centre = frames[name][:3, :3] @ com + frames[name][:3, 3]
                torque += axis @ np.cross(centre - point, mass * GRAVITY)
            worst[j] = max(worst[j], abs(torque))
    return [float(max(floor, 10 * math.ceil(EFFORT_MARGIN * value / 10)))
            for value, floor in zip(worst, EFFORT_FLOOR_NM)]


def mount_elements():
    plate = ET.Element("link", name=PREFIX + "mount_plate")
    size = np.array(PLATE_SIZE)
    mass = 2700. * size.prod()
    inertia = np.diag([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2]) * mass / 12
    plate.append(inertial_element(mass, (0., 0., size[2] / 2), inertia))
    for tag in ("visual", "collision"):
        node = ET.SubElement(plate, tag)
        node.append(ET.Element("origin", xyz=f"0 0 {size[2] / 2:g}", rpy="0 0 0"))
        ET.SubElement(ET.SubElement(node, "geometry"), "box", size=" ".join(f"{v:g}" for v in size))
        if tag == "visual":
            ET.SubElement(ET.SubElement(node, "material", name=PREFIX + "mount_plate"), "color", rgba="0.12 0.14 0.17 1")
    joints = []
    for name, parent, child, xyz in (("mount_joint", "base_link", PREFIX + "mount_plate", PLATE_BOTTOM),
                                     ("base_joint", PREFIX + "mount_plate", PREFIX + "base_link", (0., 0., size[2]))):
        joint = ET.Element("joint", name=PREFIX + name, type="fixed")
        joint.append(ET.Element("origin", xyz=" ".join(f"{v:g}" for v in xyz), rpy="0 0 0"))
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        joints.append(joint)
    return plate, joints, mass


def compose():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tcp_in_torch, torch_com, torch_inertia = torch_frames()
    b2, dropped = b2_elements()
    arm_links, arm_joints, cad_mass, efforts = arm_elements(torch_com, torch_inertia)
    plate, mount_joints, plate_mass = mount_elements()
    robot = ET.Element("robot", name=NAME)
    robot.extend([*(e for e in b2 if e.tag == "link"), plate, *arm_links,
                  *(e for e in b2 if e.tag == "joint"), *mount_joints, *arm_joints])
    ET.indent(robot, space="  ")
    urdf = WORK_DIR / f"{NAME}.urdf"
    ET.ElementTree(robot).write(urdf, encoding="utf-8", xml_declaration=True)

    end = next(j for j in arm_joints if j.get("name") == PREFIX + "end_joint")
    torch = next(j for j in arm_joints if j.get("name") == PREFIX + "welding_torch_fixed_joint")
    link6_from_tcp = origin_matrix(end.find("origin")) @ origin_matrix(torch.find("origin")) @ tcp_in_torch
    # The physics collider is this hull; planners check it against observed faces.
    vertices = stl_vertices(E10L_DIR / "meshes/welding_torch.STL").reshape(-1, 3)
    hull = vertices[ConvexHull(vertices).vertices]
    hull_tcp = (np.linalg.inv(tcp_in_torch) @ np.c_[hull, np.ones(len(hull))].T).T[:, :3]
    trunk_from_arm_base = np.eye(4)
    trunk_from_arm_base[:3, 3] = np.add(PLATE_BOTTOM, (0., 0., PLATE_SIZE[2]))
    unitree = subprocess.run(["git", "-C", str(B2_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "asset": f"{NAME}/{NAME}.usda",
        "units": "metres",
        "up_axis": "Z",
        "base": "floating",
        "sources": {
            "b2": {"url": "https://github.com/unitreerobotics/unitree_ros", "path": "robots/b2_description",
                   "commit": unitree.stdout.strip() or None},
            "e10l": {"archive": E10L_ARCHIVE.name,
                     "sha256": hashlib.sha256(E10L_ARCHIVE.read_bytes()).hexdigest() if E10L_ARCHIVE.is_file() else None},
        },
        "dropped_b2_links": sorted(dropped),
        "leg_joints": [j.get("name") for j in b2 if j.tag == "joint" and j.get("type") == "revolute"],
        "arm_joints": [j.get("name") for j in arm_joints if j.get("type") == "revolute"],
        "arm_base_link": PREFIX + "base_link",
        "tool_link": PREFIX + "Link6",
        "trunk_from_arm_base": trunk_from_arm_base.tolist(),
        "link6_from_tcp": link6_from_tcp.tolist(),
        "torch_hull_tcp": "torch_hull_tcp.npy",
        "mount_plate": {"size_m": list(PLATE_SIZE), "bottom_in_trunk_m": list(PLATE_BOTTOM), "mass_kg": plate_mass},
        "assumptions": {
            "arm_mass_kg": ARM_MASS_KG, "arm_cad_mass_kg": cad_mass, "torch_mass_kg": TORCH_MASS_KG,
            "arm_effort_nm": efforts, "effort_margin": EFFORT_MARGIN, "effort_floor_nm": list(EFFORT_FLOOR_NM),
            "arm_velocity_deg_s": list(JOINT_SPEED_DEG_S),
            "tcp": "wire tip of the torch mesh, +Z along the straight nozzle",
        },
    }
    return urdf, manifest, hull_tcp


def convert(urdf):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        extensions = omni.kit.app.get_app().get_extension_manager()
        extensions.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        shutil.rmtree(OUTPUT_DIR / NAME, ignore_errors=True)
        config = URDFImporterConfig(urdf_path=str(urdf), usd_path=str(OUTPUT_DIR), fix_base=False,
                                    allow_self_collision=False)
        # Kit's fast shutdown can end the process inside close(), so report first.
        print(f"USD: {URDFImporter(config).import_urdf()}", flush=True)
    finally:
        app.close(wait_for_replicator=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf-only", action="store_true", help="Compose the URDF without the Isaac Sim conversion")
    args = parser.parse_args()
    urdf, manifest, hull_tcp = compose()
    print(f"URDF: {urdf}")
    print(json.dumps(manifest["assumptions"], indent=2))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    np.save(OUTPUT_DIR / manifest["torch_hull_tcp"], hull_tcp.astype(np.float32))
    if not args.urdf_only:
        convert(urdf)


if __name__ == "__main__":
    main()
