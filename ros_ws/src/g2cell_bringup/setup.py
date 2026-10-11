from glob import glob

from setuptools import setup

package_name = "g2cell_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "check_stack = g2cell_bringup.check_stack:main",
        "g2cell_engine_isaacsim.py = g2cell_bringup.engine_objrelay:main",
    ]},
)
