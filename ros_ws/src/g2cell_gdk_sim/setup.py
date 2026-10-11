from setuptools import setup

package_name = "g2cell_gdk_sim"

setup(
    name=package_name,
    version="0.1.0",
    # The importable module is agibot_gdk (the GDK's own name), not the ROS package name.
    packages=["agibot_gdk"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "NOTICE", "sim.sh", "run_training_solutions.sh"]),
        ("share/" + package_name + "/scripts", ["scripts/offline_smoke.py", "scripts/head_camera_check.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    license="Apache-2.0",
)
