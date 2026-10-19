"""Named local launcher configurations and portable single-configuration files."""

import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class ConfigurationLibrary:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self, defaults: Path) -> None:
        if not self.path.exists():
            profiles = ConfigurationLibrary(defaults).entries()
            write_json(self.path, {'version': 1, 'profiles': profiles})

    def entries(self) -> dict:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('profiles'), dict):
            raise ValueError('配置库格式无效')
        return data['profiles']

    def save(self, name: str, values: dict) -> None:
        name = name.strip()
        if not name:
            raise ValueError('配置名称不能为空')
        profiles = self.entries()
        profiles[name] = {key: value for key, value in values.items() if key != 'api_key'}
        write_json(self.path, {'version': 1, 'profiles': profiles})

    def delete(self, name: str) -> None:
        profiles = self.entries()
        del profiles[name]
        write_json(self.path, {'version': 1, 'profiles': profiles})

    @staticmethod
    def export(path: Path, name: str, values: dict) -> None:
        write_json(path, {'format': 'ninfer-launcher-config', 'version': 1, 'name': name,
                          'settings': {key: value for key, value in values.items() if key != 'api_key'}})

    @staticmethod
    def read_export(path: Path) -> tuple[str, dict]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(data, dict) or data.get('format') != 'ninfer-launcher-config' or
                data.get('version') != 1 or not isinstance(data.get('name'), str) or
                not data['name'].strip() or not isinstance(data.get('settings'), dict)):
            raise ValueError('请选择 NInfer 配置 JSON 文件')
        values = {key: value for key, value in data['settings'].items() if key != 'api_key'}
        return data['name'].strip(), values
