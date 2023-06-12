from setuptools import find_packages, setup

setup(
    name="sovitssvc",
    version="0.0.1",
    description="so vits svc for jsut train",
    author="Kai Washizaki",
    author_email="bandad.kw@gmail.com",
    long_description_content_type="text/markdown",
    package_data={"": ["_example_data/*"]},
    packages=find_packages(),
    include_package_data=True,
)