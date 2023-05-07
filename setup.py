import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='final_project',
    version='0.0.1',
    author='Parker DiPaolo, Sophie Henry, Kris Cook, Sydney Toltz',
    author_email='pd702@georgetown.edu, sgh60@georgetown.edu, klc132@georgetown.edu, stt25@georgetown.edu',
    description='Final project code',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
