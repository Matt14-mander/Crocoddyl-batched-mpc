"""Build the isolated Z1 control asset from the pinned upstream files (stdlib only).

This is an offline asset preparation script, not part of the Go2 runtime path.
Run with Python -B. The gripper is fixed at jointGripper=0 rad; no Go2 mounting
transform, payload, TCP offset or hardware actuator rating is invented here.
"""

import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def build():
    directory = Path(__file__).resolve().parent
    source = json.loads((directory / "SOURCE.json").read_text())
    for item in source["files"]:
        data = (directory / item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        assert hashlib.sha1(blob).hexdigest() == item["git_blob_sha1"], item["path"]

    upstream = directory / "upstream/robots/z1_description/xacro"
    original = ET.parse(upstream / "z1.urdf").getroot()
    xacro = ET.parse(upstream / "robot.xacro").getroot()
    constants = ET.parse(upstream / "const.xacro").getroot()
    namespace = "{http://www.ros.org/wiki/xacro}"
    properties = {
        node.attrib["name"]: node.attrib["value"]
        for node in constants.findall(namespace + "property")
    }
    output = ET.Element("robot", {"name": "z1_fixed_gripper"})
    for node in original:
        if node.tag == "link" and node.attrib["name"] != "world":
            output.append(copy.deepcopy(node))
        elif node.tag == "joint" and node.attrib["name"] != "base_static_joint":
            output.append(copy.deepcopy(node))

    gripper = next(
        node
        for node in xacro.findall(namespace + "if")
        if node.attrib["value"] == "$(arg UnitreeGripper)"
    )
    for node in gripper:
        if node.tag not in ("link", "joint"):
            continue
        node = copy.deepcopy(node)
        if node.tag == "joint" and node.attrib["name"] == "jointGripper":
            node.set("type", "fixed")
            for child in list(node):
                if child.tag in ("axis", "limit", "dynamics"):
                    node.remove(child)
        for element in node.iter():
            for key, value in element.attrib.items():

                def substitute(match):
                    literal = properties[match.group(1)]
                    float(literal)  # Only numeric literals; no general xacro evaluation.
                    return literal

                value = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", substitute, value)
                assert "${" not in value and "$(" not in value, value
                element.set(key, value)
        output.append(node)

    for node in output:
        if node.tag == "joint" and node.attrib["name"] == "gripperStator":
            node.set("name", "gripper_mount_joint")
        elif node.tag == "joint" and node.attrib["name"] == "jointGripper":
            node.set("name", "gripper_lock_joint")
        node.set("name", "z1_" + node.attrib["name"])
        if node.tag == "joint":
            for tag in ("parent", "child"):
                frame = node.find(tag)
                frame.set("link", "z1_" + frame.attrib["link"])
    joints = output.findall("joint")
    assert [j.attrib["name"] for j in joints if j.attrib["type"] == "revolute"] == [
        f"z1_joint{i}" for i in range(1, 7)
    ]
    links = {node.attrib["name"] for node in output.findall("link")}
    children = {node.find("child").attrib["link"] for node in joints}
    assert links - children == {"z1_link00"}
    assert len(links) == 9 and len(joints) == 8
    ET.indent(output, space="  ")
    target = directory / "z1_fixed_gripper.urdf"
    ET.ElementTree(output).write(target, encoding="utf-8", xml_declaration=True)
    with target.open("ab") as stream:
        stream.write(b"\n")
    print(target.name + " sha256=" + hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
