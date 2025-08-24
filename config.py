
from configparser import ConfigParser


config = ConfigParser()

config.read("config.ini")


def get_config(section: str, option: str, fallback=None):
    return config.get(section, option, fallback=fallback)