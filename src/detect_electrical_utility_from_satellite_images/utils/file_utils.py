import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def drop_jpg_paths_with_no_npz_pair(jpg_paths: list[Path]) -> list[Path]:
    new_list = []
    for jpg_path in jpg_paths:
        if jpg_path.with_suffix(".npz").exists():
            logger.debug(f"npz file for {jpg_path.name} exists")
            new_list.append(jpg_path)
        else:
            logger.warning(f"npz file for {jpg_path} does not exist. it is ignored")
    return new_list
