from setuptools import setup

package_name = "g2cell_learning"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/configs", ["configs/server_a6000.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "h5py", "torch", "torchvision", "pyarrow", "av",
                      "matplotlib"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "g2_export_lerobot = g2cell_learning.export_lerobot:main",
        "g2_train = g2cell_learning.train:main",
        "g2_eval_offline = g2cell_learning.eval_offline:main",
        "g2_dataset_card = g2cell_learning.dataset_card:main",
        "g2_prepare_cache = g2cell_learning.prepare_cache:main",
    ]},
)
