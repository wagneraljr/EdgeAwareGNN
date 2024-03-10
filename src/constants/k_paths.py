from pathlib import Path

class KPaths:
    root_project: str = str(Path(__file__).parent.parent.parent) + '/'
    path_checkcpoints: str = root_project + 'checkpoints/'
    path_graphics: str = root_project + 'graphics/'
    path_logs: str = root_project + 'logs/'
    path_data: str = root_project + 'data/'
    path_results: str = root_project + 'train_results/'
    path_hyperparameters: str = root_project + 'hyperparameters_config/'        