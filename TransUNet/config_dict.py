"""Minimal ConfigDict replacement for ml_collections (TransUNet configs)."""


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[name] = value
