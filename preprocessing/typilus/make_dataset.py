import os.path

import docker
from ncc.utils.logging import LOGGER
from docker import errors
from ncc import __NCC_DIR__

IMAGE_IDENTIFIER = "3usi9/typilus-env"


def from_raw(raw_path: str,
             store_path):
    if(not os.path.exists(raw_path)):
        raise FileNotFoundError("The specified Raw Path is invalid.")
    if(not os.path.exists(store_path)):
        os.mkdir(store_path)
    LOGGER.info(f"All the python source code in '{raw_path}' will be converted to graph and stored to '{store_path}'.")
    client = docker.from_env()
    try:
        client.images.get(IMAGE_IDENTIFIER)
    except(docker.errors.ImageNotFound):
        LOGGER.info("The typilus docker image not found, pulling...")
        # client.images.pull(IMAGE_IDENTIFIER)
        os.system("docker pull " + IMAGE_IDENTIFIER)
    LOGGER.info(f"Generating the typilus graph for '{raw_path}'")
    container = client.containers.run(IMAGE_IDENTIFIER, command='bash scripts/prepare_data_from_existings.sh /mnt',
                                      volumes=[raw_path + ':/mnt',
                                               store_path + ':/usr/data'],
                                      detach=True,
                                      remove=True)
    output = container.attach(stdout=True, stream=True, logs=True)
    for line in output:
        print(line)
    LOGGER.info(f"The typilus graph for all python sources in '{raw_path}' is successfully generated.")

