from setuptools import setup

package_name = "g2cell_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "pick_place_demo = g2cell_demo.demo_node:main",
        "reach_check = g2cell_demo.reach_check:main",
        "record_cams = g2cell_demo.record_cams:main",
        "drive_in = g2cell_demo.drive_in:main",
        "packing_demo = g2cell_demo.packing_demo:main",
    ]},
)
