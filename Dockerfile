# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM python:3.12.0-slim AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# Use Docker-in-Docker image
FROM docker:latest

# Install dependencies
RUN apk add --no-cache \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    sudo

# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----
FROM python:3.11.0-slim AS python-mrd-devcontainer

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxslt1.1 \
    libhdf5-dev \
    libboost-program-options-dev \
    libpugixml-dev \
    dos2unix \
    nano \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/

# Install Python dependencies and check if installation succeeds
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip freeze

RUN mkdir -p /opt/code

# Install necessary dependencies
RUN apt update && apt install -y git git-lfs && git lfs install

# Tell nano to remember its position from the last time it opened a file
RUN echo "set positionlog" > ~/.nanorc

# Python MRD library
RUN pip3 install h5py==3.10.0 ismrmrd==1.14.1

# Clone additional repositories
RUN mkdir -p /opt/code && \
    cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir . && \
    pip freeze
    
# Clone automated-fetal-mri and pull LFS data
RUN git clone https://github.com/saranevessilva/automated-fetal-mri.git /opt/code/automated-fetal-mri && \
    cd /opt/code/automated-fetal-mri && \
    git lfs install && \
    git lfs pull

# Copy the 'eagle' folder from automated-fetal-mri into the main folder
# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM python:3.12.0-slim AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# Use Docker-in-Docker image
FROM docker:latest

# Install dependencies
RUN apk add --no-cache \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    sudo

# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----
FROM python:3.11.0-slim AS python-mrd-devcontainer

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxslt1.1 \
    libhdf5-dev \
    libboost-program-options-dev \
    libpugixml-dev \
    dos2unix \
    nano \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/

# Install Python dependencies and check if installation succeeds
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip freeze

RUN mkdir -p /opt/code

# Install necessary dependencies
RUN apt update && apt install -y git git-lfs && git lfs install

# Tell nano to remember its position from the last time it opened a file
RUN echo "set positionlog" > ~/.nanorc

# Python MRD library
RUN pip3 install h5py==3.10.0 ismrmrd==1.14.1

# Clone additional repositories
RUN mkdir -p /opt/code && \
    cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir . && \
    pip freeze
    
# Clone automated-fetal-mri and pull LFS data
RUN git clone https://github.com/saranevessilva/automated-fetal-mri.git /opt/code/automated-fetal-mri && \
    cd /opt/code/automated-fetal-mri && \
    git lfs install && \
    git lfs pull

# Copy the 'eagle' folder from automated-fetal-mri into the main folder

RUN mkdir -p /opt/code/python-ismrmrd-server && \
    cp -r /opt/code/automated-fetal-mri/eagle /opt/code/python-ismrmrd-server/eagle
    
# RUN cp -r /opt/code/automated-fetal-mri/eagle /opt/code/python-ismrmrd-server/eagle

RUN rm -rf /opt/code/automated-fetal-mri

# matplotlib is used by rgb.py and provides various visualization tools including colormaps
# pydicom is used by dicom2mrd.py to parse DICOM data
RUN pip3 install --no-cache-dir matplotlib==3.8.2 pydicom==3.0.1

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y dcm2niix
    
# ----- 3. Copy deployed code into the devcontainer for deployment -----
FROM python-mrd-devcontainer AS python-mrd-runtime

# If building from the GitHub repo, uncomment the below section, open a command
# prompt in the folder containing this Dockerfile and run the command:
#    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN cd /opt/code && \
#     git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the python-ismrmrd-server folder, uncomment the following lines
# below and run the command:
#    docker build --no-cache -t fire-python-custom -f docker/Dockerfile ./
# RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

# Throw an explicit error if docker build is run from the folder *containing*
# python-ismrmrd-server instead of within it (i.e. old method)
RUN if [ -d /opt/code/python-ismrmrd-server/python-ismrmrd-server ]; then echo "docker build should be run inside of python-ismrmrd-server instead of one directory up"; exit 1; fi

# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

# CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=invertcontrast"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
RUN rm -rf /opt/code/automated-fetal-mri

# matplotlib is used by rgb.py and provides various visualization tools including colormaps
# pydicom is used by dicom2mrd.py to parse DICOM data
RUN pip3 install --no-cache-dir matplotlib==3.8.2 pydicom==3.0.1

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y dcm2niix
    
# ----- 3. Copy deployed code into the devcontainer for deployment -----
FROM python-mrd-devcontainer AS python-mrd-runtime

# If building from the GitHub repo, uncomment the below section, open a command
# prompt in the folder containing this Dockerfile and run the command:
#    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN cd /opt/code && \
#     git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the python-ismrmrd-server folder, uncomment the following lines
# below and run the command:
#    docker build --no-cache -t fire-python-custom -f docker/Dockerfile ./
# RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

# Throw an explicit error if docker build is run from the folder *containing*
# python-ismrmrd-server instead of within it (i.e. old method)
RUN if [ -d /opt/code/python-ismrmrd-server/python-ismrmrd-server ]; then echo "docker build should be run inside of python-ismrmrd-server instead of one directory up"; exit 1; fi

# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

# CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=invertcontrast"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
