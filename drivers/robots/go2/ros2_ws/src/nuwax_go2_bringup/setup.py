from setuptools import find_packages, setup
from glob import glob
import os


package_name = "nuwax_go2_bringup"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nuwax_robot_bridge",
    maintainer_email="ml@example.com",
    description="Go2 端侧 ROS2 工作空间启动包",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
