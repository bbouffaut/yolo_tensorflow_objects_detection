cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/src/opencv_contrib-3.3.0/modules \
    -D PYTHON_EXECUTABLE=~/virtualenv_tensorflow_python3/bin/python3 \
    -D BUILD_EXAMPLES=ON ..
