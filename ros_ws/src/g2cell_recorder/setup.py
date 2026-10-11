from setuptools import setup

package_name = "g2cell_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("lib/" + package_name, ["scripts/record_episode.sh"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "bag_to_hdf5 = g2cell_recorder.bag_to_hdf5:main",
        "mark_episode = g2cell_recorder.mark_episode:main",
        "throttle_relay = g2cell_recorder.throttle_relay:main",
    ]},
)
