# Kill the adb daemon server so that the container can connect to the devices
adb-kill server 2> /dev/null
# Build the docker image
docker build -t snpe .
read -p 'Press a key to run docker (or ctrl+c to stop) ' input
# Run it with access to the usb devices for adb
docker run -v "$(pwd)"/mount:/mnt/files --privileged -v /dev/bus/usb:/dev/bus/usb -it snpe $@
