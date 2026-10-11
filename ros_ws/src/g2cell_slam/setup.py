from glob import glob

from setuptools import setup

package_name = "g2cell_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/maps", glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "cmd_vel_bridge = g2cell_slam.cmd_vel_bridge:main",
        "cloud_merger = g2cell_slam.cloud_merger:main",
        "drive_path = g2cell_slam.drive_path:main",
        "nav_goal = g2cell_slam.nav_goal:main",
    ]},
)
