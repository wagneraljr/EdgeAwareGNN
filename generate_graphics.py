import glob
import json
import os
import sys

from colorama import Fore, Style
from src.constants.k_paths import KPaths
from src.utils.calculations_util import CalculationsUtil
from src.utils.data_utils import load_data
from src.utils.plot_utils import PlotUtils

def generate_graphics(period):
    json_files = []
    all_data = {}
    path_to_results = KPaths.path_results+f"{period}/"
    for arquivo in glob.glob(os.path.join(path_to_results, '*.json')):
        json_files.append(arquivo)
        
    markers = {}
    markers_list = ['o','s','^']


    for i,json_file in enumerate(json_files):
        with open(json_file,'r') as file:
            data = json.load(file)
        model_name,period = json_file.replace(path_to_results,"").split("_train_results_")
        period = period.replace(".json","")
        all_data[model_name] = data 
        markers[model_name] = markers_list[i]

    #TODO: gerar markers de acordo com o numero de modelos 

    gml_file = KPaths.path_data + "abilene/Abilene.gml"
    tm_test = KPaths.path_data + "abilene/target/test/tm.2004-09-10.16-00-00.dat"
    node_features, edge_indices, edge_features, actual_node_loads = load_data(gml_file, tm_test)

    actual_node_loads = actual_node_loads.detach().numpy()
    model_names = []
    losses_dict = {}
    rs = {}
    predictions_dict ={}
    metrics_dict = {}
    for i,(model_name,results) in enumerate(all_data.items()):
        model_names.append(model_name)
        losses_dict[model_name] = results['losses']
        rs[model_name] = markers_list[i]
        prediction = results['predictions']
        predictions_dict[model_name] = prediction
        metrics_dict[model_name] = CalculationsUtil.compute_metrics(actual_node_loads, prediction)


    PlotUtils.plot_loss_curves(losses_dict, markers,KPaths.path_graphics+'loss/',f"loss_curves_{period}.png")

    PlotUtils.plot_predicted_vs_actual(predictions_dict, actual_node_loads,KPaths.path_graphics+'predicted_vs_target/',
                                    f"predicted_vs_actual_{period}.png")
    r2s = PlotUtils.plot_multi_r2(actual_node_loads,list(predictions_dict.values()),model_names,KPaths.path_graphics+'r2/',
                            f'r2_{period}.png')
    for i,r2 in enumerate(r2s):
        print(f"{model_names[i]} R2: {r2s[i]:.2f}")
        
    PlotUtils.plot_metrics(metrics_dict, KPaths.path_graphics + 'metrics/',f'metrics_{period}.png')

if __name__ == "__main__":
    args = sys.argv
    period = args[1]
    print()
    print(f"Generating graphics - Type: {period}")
    generate_graphics(period)
    print(Fore.GREEN + 'Completed\n' + Style.RESET_ALL)
    print()

