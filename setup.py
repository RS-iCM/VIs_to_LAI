"""
Setup script for VIs_to_LAI_crops package
Supports RUN_Python_Rice.ipynb, RUN_Python_Barley.ipynb, RUN_Python_Wheat.ipynb, 
RUN_Python_Maize.ipynb, RUN_Python_LAI_2D_Rice.ipynb, and RUN_Python_LAI_2D_Maize.ipynb
"""

from setuptools import setup, find_packages
import os

# Get the long description from README if available
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='vis-to-lai-crops',
    version='1.0.0',
    description='Simulate LAI (Leaf Area Index) from Vegetation Indices (VIs) for crops',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='J Ko and Tim Ng',
    author_email='',
    url='',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'pyyaml>=5.4.0',
        'tensorflow>=2.8.0',
        'keras>=2.8.0',
        'h5py>=3.0.0',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'notebook>=6.4.0',
        ],
        '2d': [
            'cartopy>=0.20.0',  # For 2D mapping and shapefile support
            'tqdm>=4.64.0',  # Progress bars
        ],
        'all': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'notebook>=6.4.0',
            'cartopy>=0.20.0',
            'tqdm>=4.64.0',
        ],
    },
    package_data={
        '': [
            '*.ipynb',
            '*.inp',  # Input configuration files for 2D
            'data/*.csv',
            'data/*.txt',
            'data/*.OBS',
            'models/*.h5',
            'models/*.pkl',
            'outputs/*.out',
            'outputs/**/*.bin',  # Binary output files for 2D
            'outputs/**/*.txt',
            'codes/*.py',
            'codes/each_crop_model/*.py',
            # 2D specific directories
            'class_map_*/*.dat',
            'class_map_*/*.bin',
            'class_map_*/*.hdr',
            'vis_*/*.bin',  # Vegetation indices binary files
            'vis_*/*.dat',
            'Shape_*/*',  # Shapefile directories
            'USA_*/*',  # USA map directories
        ],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Agricultural Science',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='agriculture LAI vegetation-indices remote-sensing crops',
)

