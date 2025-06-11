from data.load_data import load_data

# opening config
import yaml
with open("config/baseline_config.yaml", "r") as f:
    config = yaml.safe_load(f)

data = load_data(config["data"]["path"])
