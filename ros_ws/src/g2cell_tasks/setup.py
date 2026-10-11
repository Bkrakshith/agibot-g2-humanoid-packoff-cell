from glob import glob

from setuptools import setup

package_name = "g2cell_tasks"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "pick_bag_server = g2cell_tasks.pick_bag_server:main",
        "cell_manager = g2cell_tasks.cell_manager:main",
    ]},
)
