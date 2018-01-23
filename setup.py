from setuptools import setup
from setuptools import find_packages

def read_readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='yolo_keras_tensorflow',
    version='1.0.0',
    license='MIT',
    description='Module to load Keras + Tensorflow YOLO implementation based on 80 classes detection',
    long_description=read_readme(),
    url='https://github.com/bbouffaut/yolo_tensorflow_objects_detection',
    author='Baptiste BOUFFAUT',
    author_email='baptiste.bouffaut@gmail.com',
    keywords='tensorflow keras Convutional Neural-Network yolo',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    packages=find_packages(),
    #packages=['faster_rcnn_tf'],
    #py_modules=['faster_rcnn_tf'],
    install_requires=[],
    test_suite='',
    tests_require=[],
    entry_points={
        'console_scripts': ['yolo_keras_tensorflow=yolo_keras_tensorflow:main'],
    },
    include_package_data=True,
    zip_safe=False,
)
